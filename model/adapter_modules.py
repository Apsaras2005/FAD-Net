import logging
from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from .MSDeformAttn import MSDeformAttn
from timm.layers import DropPath

_logger = logging.getLogger(__name__)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape

    spatial_shapes_py = [
        (h // 8, w // 8),
        (h // 16, w // 16),
        (h // 32, w // 32)
    ]

    spatial_shapes_injector = torch.as_tensor(spatial_shapes_py, dtype=torch.long, device=x.device)

    level_start_index_injector = torch.cat((spatial_shapes_injector.new_zeros(
        (1,)), spatial_shapes_injector.prod(1).cumsum(0)[:-1]))

    reference_points_injector = get_reference_points(spatial_shapes_py, x.device)

    deform_inputs1 = [reference_points_injector, spatial_shapes_injector, level_start_index_injector]

    return deform_inputs1

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        return x


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=4, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        def _inner_forward(query, feat):
            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        return query


class DualExtractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False,
                 gate_initial_value=0.5):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm_x = norm_layer(dim)
        self.feat_norm_y = norm_layer(dim)
        self.attn_x = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                   n_points=n_points, ratio=deform_ratio)
        self.attn_y = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                   n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        self.gate_x = nn.Parameter(torch.full((1,), gate_initial_value))
        self.gate_y = nn.Parameter(torch.full((1,), gate_initial_value))
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, c, x_vit, y_vit, H_q, W_q, H_kv, W_kv):
        def _inner_forward(c, x_vit, y_vit, H_q, W_q, H_kv, W_kv):
            # 1. K/V (ViT) feature spatial shapes
            spatial_shapes = torch.as_tensor([(H_kv, W_kv)], dtype=torch.long, device=c.device)
            level_start_index = torch.as_tensor([0], dtype=torch.long, device=c.device)

            query_spatial_shapes = [(H_q, W_q)]
            reference_points = get_reference_points(query_spatial_shapes, c.device)

            query = self.query_norm(c)
            update_x = self.attn_x(query, reference_points, self.feat_norm_x(x_vit), spatial_shapes, level_start_index,
                                   None)
            update_y = self.attn_y(query, reference_points, self.feat_norm_y(y_vit), spatial_shapes, level_start_index,
                                   None)

            c_intermediate = c + self.gate_x * update_x + self.gate_y * update_y

            if self.with_cffn:
                c_final = c_intermediate + self.drop_path(self.ffn(self.ffn_norm(c_intermediate), H_q, W_q))
            else:
                c_final = c_intermediate
            return c_final

        if self.with_cp and c.requires_grad:
            c = cp.checkpoint(_inner_forward, c, x_vit, y_vit, H_q, W_q, H_kv, W_kv)
        else:
            c = _inner_forward(c, x_vit, y_vit, H_q, W_q, H_kv, W_kv)
        return c


class InjectionBlock(nn.Module):
    def __init__(self, dim, norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False, **kwargs):
        super().__init__()
        deform_num_heads = kwargs.get('DEFORM_NUM_HEADS', 8)
        n_points = kwargs.get('N_POINTS', 4)
        init_values = kwargs.get('INIT_VALUES', 0.0)
        deform_ratio = kwargs.get('DEFORM_RATIO', 1.0)

        self.injector_x = Injector(dim=dim, n_levels=3, num_heads=deform_num_heads, init_values=init_values,
                                   n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                   with_cp=with_cp)
        self.injector_y = Injector(dim=dim, n_levels=3, num_heads=deform_num_heads, init_values=init_values,
                                   n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                   with_cp=with_cp)

    def forward(self, x_vit, y_vit, c, deform_inputs_injector):
        ref_points, spat_shapes, lvl_start_idx = deform_inputs_injector

        x_vit_injected = self.injector_x(query=x_vit, reference_points=ref_points,
                                         feat=c, spatial_shapes=spat_shapes,
                                         level_start_index=lvl_start_idx)

        y_vit_injected = self.injector_y(query=y_vit, reference_points=ref_points,
                                         feat=c, spatial_shapes=spat_shapes,
                                         level_start_index=lvl_start_idx)

        return x_vit_injected, y_vit_injected



class ExtractionBlock(nn.Module):
    def __init__(self, dim, norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False, **kwargs):
        super().__init__()
        deform_num_heads = kwargs.get('DEFORM_NUM_HEADS', 8)
        n_points = kwargs.get('N_POINTS', 4)
        deform_ratio = kwargs.get('DEFORM_RATIO', 1.0)
        use_extra_extractor = kwargs.get('USE_EXTRA_EXTRACTOR', True)
        with_cffn = kwargs.get('WITH_CFFN', True)
        cffn_ratio = kwargs.get('CFFN_RATIO', 0.25)
        gate_initial_value = kwargs.get('GATE_INITIAL_VALUE', 0.5)

        self.extractor = DualExtractor(
            dim=dim, n_levels=1, num_heads=deform_num_heads, n_points=n_points,
            norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
            cffn_ratio=cffn_ratio, with_cp=with_cp,
            gate_initial_value=gate_initial_value
        )

        if use_extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                DualExtractor(dim=dim, num_heads=deform_num_heads, n_points=n_points, norm_layer=norm_layer,
                              with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                              with_cp=with_cp, gate_initial_value=gate_initial_value)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, c, x_vit, y_vit, H_c, W_c, H_vit, W_vit):
        c_flat = c.flatten(2).transpose(1, 2)

        c_updated_flat = self.extractor(c=c_flat, x_vit=x_vit, y_vit=y_vit,
                                        H_q=H_c, W_q=W_c, H_kv=H_vit, W_kv=W_vit)

        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c_updated_flat = extractor(c=c_updated_flat, x_vit=x_vit, y_vit=y_vit,
                                           H_q=H_c, W_q=W_c, H_kv=H_vit, W_kv=W_vit)

        B, _, C = c_updated_flat.shape
        c_updated = c_updated_flat.transpose(1, 2).view(B, C, H_c, W_c)

        return c_updated


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes), nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes), nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 8 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(8 * inplanes), nn.ReLU(inplace=True)
        ])
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(8 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
            return c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs