import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, groups=1, norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups,
                              bias=norm_layer is None)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act = act_layer() if act_layer else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels, norm_layer, act_layer, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        for pool_scale in self.pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    _ConvModule(
                        in_channels,
                        channels,
                        1,
                        norm_layer=norm_layer,
                        act_layer=act_layer
                    )
                )
            )

    def forward(self, x):
        ppm_outs = []
        for ppm_layer in self:
            ppm_out = ppm_layer(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UPerNetHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, pool_scales=(1, 2, 3, 6),
                 dropout_ratio=0.1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, align_corners=False):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners

        self.psp_module = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            norm_layer=norm_layer,
            act_layer=act_layer,
            align_corners=self.align_corners)

        self.bottleneck = _ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            norm_layer=norm_layer,
            act_layer=act_layer)

        self.lateral_convs = nn.ModuleList()
        for in_ch in self.in_channels[:-1]:
            self.lateral_convs.append(
                _ConvModule(in_ch, self.channels, 1, norm_layer=norm_layer, act_layer=act_layer)
            )

        num_fusion_stages = len(self.in_channels) - 1

        self.details_processors = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_fusion_stages)
        ])

        self.fusion_refiners = nn.ModuleList([
            _ConvModule(self.channels, self.channels, 3, padding=1, norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(num_fusion_stages)
        ])

        self.fpn_bottleneck = _ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            norm_layer=norm_layer,
            act_layer=act_layer)

        self.cls_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        )

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_module(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs):
        with torch.amp.autocast(device_type='cuda', enabled=False):
            inputs = [inp.float() for inp in inputs]
            laterals = [
                lateral_conv(inputs[i])
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]
            p_4 = self.psp_forward(inputs)
            p_feats = [p_4]

            for i in range(len(self.in_channels) - 2, -1, -1):
                l_shallow = laterals[i]
                p_deep = p_feats[0]

                p_deep_up = F.interpolate(
                    p_deep,
                    size=l_shallow.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners
                )

                freq_l = torch.fft.fft2(l_shallow, norm='ortho')
                freq_l_shifted = torch.fft.fftshift(freq_l, dim=(-2, -1))
                amp_l = torch.abs(freq_l_shifted)

                freq_p = torch.fft.fft2(p_deep_up, norm='ortho')
                freq_p_shifted = torch.fft.fftshift(freq_p, dim=(-2, -1))
                phase_p = torch.angle(freq_p_shifted)

                new_freq_shifted = torch.polar(amp_l, phase_p)
                new_freq = torch.fft.ifftshift(new_freq_shifted, dim=(-2, -1))
                details = torch.fft.ifft2(new_freq, norm='ortho')

                details_real = torch.real(details)
                details_processed = self.details_processors[i](details_real)

                fused_feature_sum = details_processed + p_deep_up + l_shallow

                p_current = self.fusion_refiners[i](fused_feature_sum)

                p_feats.insert(0, p_current)

            p_1_size = p_feats[0].shape[2:]
            for i in range(1, len(p_feats)):
                p_feats[i] = F.interpolate(
                    p_feats[i],
                    size=p_1_size,
                    mode='bilinear',
                    align_corners=self.align_corners
                )

            fpn_outs = torch.cat(p_feats, dim=1)
            output = self.fpn_bottleneck(fpn_outs)
            output = self.cls_seg(output)

            return output