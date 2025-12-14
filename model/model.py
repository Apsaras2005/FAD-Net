import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from .swin_transformer_v2_mona import BasicLayer, PatchEmbed, PatchMerging
from .adapter_modules import (SpatialPriorModule, InjectionBlock,
                              ExtractionBlock, deform_inputs, get_reference_points)
from .Decoder import UPerNetHead

class ViTOutputFusion(nn.Module):
    def __init__(self, channels):
        super(ViTOutputFusion, self).__init__()
        self.fusion_block = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        # x, y shape: [B, C, H, W]
        fused = torch.cat([x, y], dim=1)
        return self.fusion_block(fused)


class DualSwinV2MonaUperNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        model_config = config['MODEL']
        data_config = config['DATA']
        adapter_config = model_config['ADAPTER']
        swin_config = model_config['SWINV2']
        interaction_config = model_config['INTERACTION']
        decoder_config = model_config['DECODER']

        self.num_classes = model_config['NUM_CLASSES']
        embed_dim = swin_config['EMBED_DIM']
        depths = swin_config['DEPTHS']
        num_heads = swin_config['NUM_HEADS']
        self.num_layers = len(depths)

        mona_gabor_filters = interaction_config.get('MONA_COMPLEX_GABOR_FILTERS', 4)

        self.patch_embed_x = PatchEmbed(img_size=data_config['IMG_SIZE'], embed_dim=embed_dim)
        self.spm_x = SpatialPriorModule(inplanes=adapter_config['CONV_INPLANE'], embed_dim=embed_dim)

        self.patch_embed_y = PatchEmbed(img_size=data_config['IMG_SIZE'], embed_dim=embed_dim)
        self.spm_y = SpatialPriorModule(inplanes=adapter_config['CONV_INPLANE'], embed_dim=embed_dim)

        self.pos_drop = nn.Dropout(p=swin_config['DROP_RATE'])

        self.cnn_fusion_modules = nn.ModuleList([
            ViTOutputFusion(embed_dim),
            ViTOutputFusion(embed_dim),
            ViTOutputFusion(embed_dim)
        ])

        dpr = [x.item() for x in torch.linspace(0, model_config['DROP_PATH_RATE'], sum(depths))]
        self.layers_x = nn.ModuleList()
        self.layers_y = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer_args = dict(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(self.patch_embed_x.patches_resolution[0] // (2 ** i_layer),
                                  self.patch_embed_x.patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=swin_config['WINDOW_SIZE'],
                mlp_ratio=swin_config['MLP_RATIO'],
                qkv_bias=swin_config['QKV_BIAS'],
                drop=swin_config['DROP_RATE'],
                attn_drop=swin_config['ATTN_DROP_RATE'],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=swin_config['USE_CHECKPOINT'],
                pretrained_window_size=swin_config['PRETRAINED_WINDOW_SIZES'][i_layer],
                cross_attn_num_heads=interaction_config['CROSS_ATTN_NUM_HEADS'],
                mona_gabor_filters=mona_gabor_filters,
                lora_config=swin_config.get('LORA', None)
            )
            self.layers_x.append(BasicLayer(**layer_args))
            self.layers_y.append(BasicLayer(**layer_args))

        self.injection_blocks = nn.ModuleList([
            InjectionBlock(dim=embed_dim, **adapter_config)
            for _ in range(self.num_layers)
        ])
        self.extraction_blocks = nn.ModuleList([
            ExtractionBlock(dim=embed_dim, **adapter_config)
            for _ in range(self.num_layers - 1)
        ])

        self.dim_proj_down = nn.ModuleList([nn.Identity()] + [
            nn.Linear(int(embed_dim * 2 ** i), embed_dim) for i in range(1, self.num_layers)
        ])
        self.dim_proj_up = nn.ModuleList([nn.Identity()] + [
            nn.Sequential(nn.Linear(embed_dim, int(embed_dim * 2 ** i)), nn.ReLU(True))
            for i in range(1, self.num_layers)
        ])
        self.dim_proj_down_extractor = nn.ModuleList([
            nn.Linear(int(embed_dim * 2 ** i), embed_dim) for i in range(self.num_layers)
        ])

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))

        self.vit_fusion_modules = nn.ModuleList([
            ViTOutputFusion(int(embed_dim * 2 ** i))
            for i in range(self.num_layers)
        ])

        decoder_in_channels = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.decode_head = UPerNetHead(
            in_channels=decoder_in_channels,
            channels=decoder_config['CHANNELS'],
            num_classes=self.num_classes,
            pool_scales=decoder_config['POOL_SCALES'],
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.SyncBatchNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_rgb, x_single):
        B, _, H_orig, W_orig = x_rgb.shape
        x_single_3c = x_single.repeat(1, 3, 1, 1)

        c2_x, c3_x, c4_x = self.spm_x(x_rgb)
        c2_y, c3_y, c4_y = self.spm_y(x_single_3c)

        fused_cnn_feats = [
            self.cnn_fusion_modules[0](c2_x, c2_y),
            self.cnn_fusion_modules[1](c3_x, c3_y),
            self.cnn_fusion_modules[2](c4_x, c4_y),
        ]

        _, cnn_spatial_shapes, cnn_level_start_index = deform_inputs(x_rgb)

        c_for_injection_list = []
        for i, feat in enumerate(fused_cnn_feats):
            c_for_injection_list.append(feat.flatten(2).transpose(1, 2) + self.level_embed[i])
        c_for_injection = torch.cat(c_for_injection_list, dim=1)

        x_vit = self.pos_drop(self.patch_embed_x(x_rgb))
        y_vit = self.pos_drop(self.patch_embed_y(x_single_3c))

        decoder_inputs = []
        for i in range(self.num_layers):
            # INJECTION
            x_vit_proj_down = self.dim_proj_down[i](x_vit)
            y_vit_proj_down = self.dim_proj_down[i](y_vit)

            H_vit, W_vit = self.layers_x[i].input_resolution
            ref_points_vit = get_reference_points([(H_vit, W_vit)], device=x_vit.device)
            deform_inputs_injector_correct = [ref_points_vit, cnn_spatial_shapes, cnn_level_start_index]

            x_vit_inj, y_vit_inj = self.injection_blocks[i](
                x_vit_proj_down, y_vit_proj_down, c_for_injection, deform_inputs_injector_correct
            )

            x_vit_injected = x_vit + self.dim_proj_up[i](x_vit_inj)
            y_vit_injected = y_vit + self.dim_proj_up[i](y_vit_inj)

            x_vit_out_stage, x_vit_next = self.layers_x[i](x_vit_injected, cross_tensor=y_vit_injected)
            y_vit_out_stage, y_vit_next = self.layers_y[i](y_vit_injected, cross_tensor=x_vit_injected)

            B_out, L_out, C_out = x_vit_out_stage.shape
            H_out, W_out = self.layers_x[i].input_resolution
            x_map = x_vit_out_stage.view(B_out, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()
            y_map = y_vit_out_stage.view(B_out, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()

            fused_vit_map = self.vit_fusion_modules[i](x_map, y_map)
            decoder_inputs.append(fused_vit_map)

            if i < self.num_layers - 1:
                current_cnn_feat = fused_cnn_feats[i]
                Hc, Wc = current_cnn_feat.shape[2:]

                x_out_proj = self.dim_proj_down_extractor[i](x_vit_out_stage)
                y_out_proj = self.dim_proj_down_extractor[i](y_vit_out_stage)

                updated_cnn_feat = self.extraction_blocks[i](
                    current_cnn_feat, x_out_proj, y_out_proj,
                    Hc, Wc, H_out, W_out
                )
                fused_cnn_feats[i] = updated_cnn_feat

                c_for_injection_list = []
                for j, feat in enumerate(fused_cnn_feats):
                    c_for_injection_list.append(feat.flatten(2).transpose(1, 2) + self.level_embed[j])
                c_for_injection = torch.cat(c_for_injection_list, dim=1)

            x_vit, y_vit = x_vit_next, y_vit_next

        seg_logits = self.decode_head(decoder_inputs)
        seg_logits = F.interpolate(seg_logits, size=(H_orig, W_orig), mode='bilinear', align_corners=False)

        return seg_logits