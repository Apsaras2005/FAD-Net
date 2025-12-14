import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch.fft
from timm.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import math
import random

class DynamicSpectrumClustering:
    def __init__(self, num_prototypes=3, elastic_damping=0.5):
        self.K = num_prototypes
        self.damping = elastic_damping
        self.cached_geometry = {}

    def run(self, magnitude_map):
        H, W = magnitude_map.shape
        device = magnitude_map.device

        geom_key = (H, W, str(device))
        if geom_key in self.cached_geometry:
            grid_y, grid_x, norm_radius_map, radius_map = self.cached_geometry[geom_key]
        else:
            y = torch.arange(H, device=device) - H // 2
            x = torch.arange(W, device=device) - W // 2
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

            grid_y = grid_y.contiguous()
            grid_x = grid_x.contiguous()

            radius_map = torch.sqrt(grid_y ** 2 + grid_x ** 2)
            max_r = math.sqrt((H // 2) ** 2 + (W // 2) ** 2)
            norm_radius_map = radius_map / (max_r + 1e-6)

            self.cached_geometry[geom_key] = (grid_y, grid_x, norm_radius_map, radius_map)

        log_mag = torch.log(1 + magnitude_map)
        pre_emphasis = 1.0 + 15.0 * norm_radius_map
        geometry_correction = 1.0 / (torch.sqrt(radius_map) + 1.0)

        clustering_weight = log_mag * pre_emphasis * geometry_correction

        center_h, center_w = H // 2, W // 2
        clustering_weight[center_h, center_w] = 0 

        min_dim = min(H, W)
        dynamic_mask_r = int(min_dim * 0.02)
        if dynamic_mask_r >= 1:
            clustering_weight[center_h - dynamic_mask_r: center_h + dynamic_mask_r + 1,
            center_w - dynamic_mask_r: center_w + dynamic_mask_r + 1] = 0

        weights_flat = clustering_weight.reshape(-1)
        coords_flat = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1).float()

        top_k = int(weights_flat.shape[0] * 0.3)
        top_k = max(top_k, self.K + 1)  
        _, top_indices = torch.topk(weights_flat, top_k)

        active_weights = weights_flat[top_indices]
        active_coords = coords_flat[top_indices]

        max_r_px = H // 2
        anchor_ratios = [0.20, 0.50, 0.80]
        anchor_radii = torch.tensor([r * max_r_px for r in anchor_ratios], device=device)

        prototypes = []
        for r_target in anchor_radii:
            angle = torch.rand((), device=device) * 2 * math.pi
            cy = r_target * torch.sin(angle)
            cx = r_target * torch.cos(angle)
            prototypes.append(torch.stack([cy, cx]))
        centroids = torch.stack(prototypes)

        num_iters = 3
        with torch.no_grad():
            for _ in range(num_iters):
                dists = torch.cdist(active_coords, centroids)
                labels = torch.argmin(dists, dim=1)

                data_centroids = []
                for k in range(self.K):
                    mask = (labels == k)
                    if mask.sum() == 0:
                        data_centroids.append(centroids[k])
                        continue

                    c_coords = active_coords[mask]
                    c_weights = active_weights[mask]
                    weighted_sum = (c_coords * c_weights.unsqueeze(1)).sum(dim=0)
                    total_weight = c_weights.sum()
                    data_centroids.append(weighted_sum / (total_weight + 1e-6))

                data_centroids = torch.stack(data_centroids)

                current_radii = torch.norm(data_centroids, dim=1) + 1e-6
                directions = data_centroids / current_radii.unsqueeze(1)
                target_positions = directions * anchor_radii.unsqueeze(1)

                centroids = (1 - self.damping) * data_centroids + self.damping * target_positions

        centroid_dists = torch.norm(centroids, dim=1)
        sorted_indices = torch.argsort(centroid_dists)
        sorted_centroids = centroids[sorted_indices]

        return sorted_centroids


class LoRALinear(nn.Module):
    def __init__(
            self,
            original_layer: nn.Linear,
            rank: int,
            alpha: float,
    ):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        self.original_layer = original_layer
        self.original_layer.requires_grad_(False)

        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)

        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return original_output + lora_output


class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x
        x = self.projector(x)

        return identity + x


class Mona(nn.Module):
    def __init__(self, in_dim, factor=4):
        super().__init__()
        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = MonaOp(64)
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax
        project1 = self.project1(x)
        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)
        return identity + project2


class MonaCrossAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, q, kv):
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_output, _ = self.attn(q_norm, kv_norm, kv_norm)
        return attn_output


class MM_Mona(nn.Module):
    def __init__(self, in_dim, proj_dim=64, num_filters_k=4, num_prototypes=3):
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.k = num_filters_k
        self.num_prototypes = num_prototypes
        self.project1 = nn.Linear(in_dim, self.proj_dim)
        self.project2 = nn.Linear(self.proj_dim, in_dim)
        self.nonlinear = F.gelu
        self.dropout = nn.Dropout(p=0.1)
        self.decision_module = nn.Sequential(
            nn.Conv2d(2 * self.proj_dim, self.proj_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.proj_dim // 2, 2, kernel_size=1),
            nn.Sigmoid()
        )

        init_params = torch.zeros(self.k, 4)
        target_freqs = torch.linspace(0.05, 0.45, self.k)
        inv_sigmoid_freq = torch.log(2 * target_freqs / (1 - 2 * target_freqs))
        init_params[:, 0] = inv_sigmoid_freq[torch.randperm(self.k)]
        init_params[:, 1].normal_(mean=-0.5, std=1.0)
        target_angles = torch.linspace(0.05, 0.95, self.k)
        inverse_sigmoid_angles = torch.log(target_angles / (1 - target_angles))
        init_params[:, 2] = inverse_sigmoid_angles[torch.randperm(self.k)]
        init_params[:, 3].normal_(mean=-0.5, std=1.0)
        self.gabor_params = nn.Parameter(init_params)

        self.cluster_algo = DynamicSpectrumClustering(num_prototypes=self.num_prototypes, elastic_damping=0.5)

        self.dw_conv_3x3 = nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=3, padding=1, groups=self.proj_dim)
        self.dw_conv_5x5 = nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=5, padding=2, groups=self.proj_dim)
        self.dw_conv_7x7 = nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=7, padding=3, groups=self.proj_dim)
        self.cross_attn_spatial = MonaCrossAttention(self.proj_dim, num_heads=8)
        self.cross_attn_freq = MonaCrossAttention(self.proj_dim, num_heads=8)
        self.fusion_conv = nn.Conv2d(self.proj_dim * 2, self.proj_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

        self.cached_grids = {}
        self.cached_filters = None
        self.cached_filters_key = None  

    def _generate_log_gabor_filters(self, H, W, device):
        current_key = (H, W, str(device))

        if not self.training and self.cached_filters is not None and self.cached_filters_key == current_key:
            return self.cached_filters

        if current_key not in self.cached_grids:
            u = torch.linspace(-0.5, 0.5, W, device=device)
            v = torch.linspace(-0.5, 0.5, H, device=device)
            v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
            self.cached_grids[current_key] = (v_grid, u_grid)

        v_grid, u_grid = self.cached_grids[current_key]

        radius = torch.sqrt(u_grid ** 2 + v_grid ** 2)
        radius[H // 2, W // 2] = 1.0

        center_freq = (torch.sigmoid(self.gabor_params[:, 0]) * 0.5).view(self.k, 1, 1)
        freq_bw = F.softplus(self.gabor_params[:, 1]).view(self.k, 1, 1) + 1e-5  # 防止带宽为0
        orientation = (torch.sigmoid(self.gabor_params[:, 2]) * torch.pi).view(self.k, 1, 1)
        angle_bw = F.softplus(self.gabor_params[:, 3]).view(self.k, 1, 1) + 1e-5  # 防止角度带宽为0

        r_safe = radius / (center_freq + 1e-6)
        radial_component = torch.exp((-(torch.log(r_safe + 1e-6)) ** 2) / (2 * freq_bw ** 2))
        radial_component = radial_component.clone()
        radial_component[:, H // 2, W // 2] = 0.0

        theta = torch.atan2(v_grid, u_grid)
        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)

        ds = sintheta * torch.cos(orientation) - costheta * torch.sin(orientation)
        dc = costheta * torch.cos(orientation) + sintheta * torch.sin(orientation)
        dtheta = torch.abs(torch.atan2(ds, dc))

        angular_component = torch.exp((-dtheta ** 2) / (2 * angle_bw ** 2))
        filters = radial_component * angular_component
        filters = filters.unsqueeze(1)

        if not self.training:
            self.cached_filters = filters
            self.cached_filters_key = current_key

        return filters

    def _compute_adaptive_fusion(self, x_proj_map, spec_input, magnitude_mean, gabor_filters, h, w):
        b, c, _, _ = x_proj_map.shape
        device = x_proj_map.device

        prototypes = self.cluster_algo.run(magnitude_mean.detach() + 1e-6)

        filtered_spectra = spec_input.unsqueeze(1) * gabor_filters
        B_f, K_f, C_f, H_f, W_f = filtered_spectra.shape
        reshaped_spectra = filtered_spectra.view(B_f * K_f, C_f, H_f, W_f)

        texture_map_complex = torch.fft.ifft2(torch.fft.ifftshift(reshaped_spectra, dim=(-2, -1)), norm='ortho')
        texture_responses = torch.real(texture_map_complex).view(B_f, K_f, C_f, H_f, W_f)

        freq_out = torch.mean(texture_responses, dim=1)

        energy_E = torch.sum(torch.square(texture_responses), dim=(2, 3, 4))  # [B, K]

        center_freq_val = torch.sigmoid(self.gabor_params[:, 0]) * 0.5
        compensation_factor = torch.sqrt(center_freq_val + 0.1).to(device)
        balanced_energy = energy_E * compensation_factor.view(1, self.k)

        r_px = center_freq_val * h
        angles = torch.sigmoid(self.gabor_params[:, 2]) * torch.pi
        f_y = r_px * torch.sin(angles)
        f_x = r_px * torch.cos(angles)
        filter_centers = torch.stack([f_y, f_x], dim=1).to(device)

        dist_F_P = torch.cdist(filter_centers, prototypes)
        temperature = 20.0
        membership_M = F.softmax(-dist_F_P / temperature, dim=1)

        e_group = torch.sum(balanced_energy.unsqueeze(2) * membership_M.unsqueeze(0), dim=1)
        total_energy = e_group.sum(dim=1, keepdim=True)
        dynamic_weights = (e_group / (total_energy + 1e-6)).detach()

        w_low = dynamic_weights[:, 0].view(b, 1, 1, 1)
        w_mid = dynamic_weights[:, 1].view(b, 1, 1, 1)
        w_high = dynamic_weights[:, 2].view(b, 1, 1, 1)

        out_3 = self.dw_conv_3x3(x_proj_map)
        out_5 = self.dw_conv_5x5(x_proj_map)
        out_7 = self.dw_conv_7x7(x_proj_map)

        spatial_out = w_high * out_3 + w_mid * out_5 + w_low * out_7

        return spatial_out, freq_out

    def forward(self, x, hw_shapes, x_other=None, is_cross_path=False):
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x_fp32 = x.float()
            x_other_fp32 = x_other.float() if x_other is not None else None

            identity_final = x_fp32
            b, n, c = x_fp32.shape
            h, w = hw_shapes

            x_norm = self.norm(x_fp32) * self.gamma + x_fp32 * self.gammax
            x_proj = self.project1(x_norm)
            x_proj_map = x_proj.reshape(b, h, w, self.proj_dim).permute(0, 3, 1, 2)

            spec_x = torch.fft.fftshift(torch.fft.fft2(x_proj_map, norm='ortho'))
            mag_x = torch.abs(spec_x).mean(dim=(0, 1))

            if is_cross_path and x_other_fp32 is not None:
                x_other_norm = self.norm(x_other_fp32) * self.gamma + x_other_fp32 * self.gammax
                x_other_proj = self.project1(x_other_norm)
                x_other_proj_map = x_other_proj.reshape(b, h, w, self.proj_dim).permute(0, 3, 1, 2)

                spec_y = torch.fft.fftshift(torch.fft.fft2(x_other_proj_map, norm='ortho'))
                mag_y = torch.abs(spec_y).mean(dim=(0, 1))

                amp_x, phase_x = torch.abs(spec_x), torch.angle(spec_x)
                amp_y, phase_y = torch.abs(spec_y), torch.angle(spec_y)
                diff_map = torch.cat([torch.abs(amp_x - amp_y), torch.abs(phase_x - phase_y)], dim=1)
                gates = self.decision_module(diff_map)
                alpha_map, beta_map = gates[:, 0:1, ...], gates[:, 1:2, ...]
                recomb1 = torch.polar(amp_x, phase_y)
                recomb2 = torch.polar(amp_y, phase_x)
                spec_fused = spec_x + alpha_map * recomb1 + beta_map * recomb2
            else:
                spec_fused = spec_x
                x_other_proj_map = None
                mag_y = None

            gabor_filters = self._generate_log_gabor_filters(h, w, x_fp32.device)

            spatial_out, freq_out = self._compute_adaptive_fusion(
                x_proj_map, spec_fused, mag_x, gabor_filters, h, w
            )

            if is_cross_path and x_other_fp32 is not None:
                spatial_out_other, freq_out_other = self._compute_adaptive_fusion(
                    x_other_proj_map, spec_y, mag_y, gabor_filters, h, w
                )

                q_s = spatial_out.flatten(2).transpose(1, 2)
                kv_s = spatial_out_other.flatten(2).transpose(1, 2)
                q_f = freq_out.flatten(2).transpose(1, 2)
                kv_f = freq_out_other.flatten(2).transpose(1, 2)

                spatial_enhanced = spatial_out + self.cross_attn_spatial(q_s, kv_s).transpose(1, 2).view(b,
                                                                                                         self.proj_dim,
                                                                                                    h, w)
                freq_enhanced = freq_out + self.cross_attn_freq(q_f, kv_f).transpose(1, 2).view(b, self.proj_dim, h, w)
            else:
                spatial_enhanced = spatial_out
                freq_enhanced = freq_out

            final_fused_map = torch.cat([spatial_enhanced, freq_enhanced], dim=1)
            final_fused_map = self.fusion_conv(final_fused_map)
            final_proj = final_fused_map.permute(0, 2, 3, 1).reshape(b, n, self.proj_dim)
            nonlinear = self.nonlinear(final_proj)
            nonlinear = self.dropout(nonlinear)
            project2 = self.project2(nonlinear)

            final_output = identity_final + project2

        return final_output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0],
                 lora_rank: int = 0, lora_alpha: float = 1.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')).permute(1, 2,
                                                                                           0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        if lora_rank > 0:
            self.qkv = LoRALinear(self.qkv, rank=lora_rank, alpha=lora_alpha)
            self.proj = LoRALinear(self.proj, rank=lora_rank, alpha=lora_alpha)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        if isinstance(self.qkv, LoRALinear):
            original_qkv = F.linear(x, self.qkv.original_layer.weight, qkv_bias)

            lora_qkv = self.qkv.lora_B(self.qkv.lora_A(x)) * self.qkv.scaling

            qkv = original_qkv + lora_qkv
        else:

            qkv = F.linear(x, self.qkv.weight, qkv_bias)


        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=x.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'




class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0,
                 use_cross_attn=False,
                 cross_attn_num_heads=8,
                 use_complex_mona: bool = False,
                 mona_gabor_filters: int = 4,
                 lora_rank: int = 0, lora_alpha: float = 1.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
            lora_rank=lora_rank, lora_alpha=lora_alpha)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if lora_rank > 0:
            self.mlp.fc1 = LoRALinear(self.mlp.fc1, rank=lora_rank, alpha=lora_alpha)
            self.mlp.fc2 = LoRALinear(self.mlp.fc2, rank=lora_rank, alpha=lora_alpha)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        self.use_complex_mona = use_complex_mona
        self.use_cross_attn = use_cross_attn

        self.my_module_1 = Mona(dim)

        if self.use_complex_mona:
            self.my_module_2 = MM_Mona(dim, num_filters_k=mona_gabor_filters)
        else:
            self.my_module_2 = Mona(dim)

    def forward(self, x, cross_tensor=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x_view = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x_view, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_view

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x_attn_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_attn_out = shifted_x
        x_attn_out = x_attn_out.view(B, H * W, C)

        x = shortcut + self.drop_path(self.norm1(x_attn_out))

        x = self.my_module_1(x, (H, W))

        identity = x
        mlp_output = self.mlp(x)
        norm_output = self.norm2(mlp_output)
        x = identity + self.drop_path(norm_output)

        if self.use_complex_mona:
            final_out = self.my_module_2(x, (H, W), x_other=cross_tensor, is_cross_path=self.use_cross_attn)
        else:
            final_out = self.my_module_2(x, (H, W))

        return final_out

    def extra_repr(self) -> str:
        mona_type_2 = "Complex" if self.use_complex_mona else "Simple"
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}, " \
               f"use_cross_attn={self.use_cross_attn}, module_1=Simple, module_2={mona_type_2}"


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(x)
        x = self.norm(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0, cross_attn_num_heads=8,
                 mona_gabor_filters: int = 4,
                 lora_config: dict = None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        lora_rank = lora_config.get('rank', 0) if lora_config else 0
        lora_alpha = lora_config.get('alpha', 1.0) if lora_config else 1.0

        self.blocks = nn.ModuleList()
        for i in range(depth):
            is_last_block_in_stage = (i == depth - 1)

            self.blocks.append(SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                use_complex_mona=is_last_block_in_stage,  
                use_cross_attn=is_last_block_in_stage,  
                cross_attn_num_heads=cross_attn_num_heads,
                mona_gabor_filters=mona_gabor_filters,
                lora_rank=lora_rank, lora_alpha=lora_alpha)
            )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, cross_tensor=None):
        x_out = x

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                def create_forward_fn(block, use_cross_attn):
                    def forward_fn(input_x, input_cross):
                        if use_cross_attn:
                            return block(input_x, input_cross)
                        else:
                            return block(input_x, None)

                    return forward_fn

                if blk.use_cross_attn:
                    x_out = checkpoint.checkpoint(create_forward_fn(blk, True), x_out, cross_tensor,
                                                  use_reentrant=False)
                else:
                    x_out = checkpoint.checkpoint(create_forward_fn(blk, False), x_out, None, use_reentrant=False)
            else:
                if blk.use_cross_attn:
                    x_out = blk(x_out, cross_tensor)
                else:
                    x_out = blk(x_out, None)

        if self.downsample is not None:
            x_down = self.downsample(x_out)
            return x_out, x_down
        else:
            return x_out, x_out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()

        lora_config = kwargs.get('lora_config', None)

        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer],
                               lora_config=lora_config)  # 传递 lora_config
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        for bly in self.layers:
            pass

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x