"""
自研模块：FocusingDiffusionPyramidNetwork (FDPN)
聚焦扩散特征金字塔网络

核心创新：
1. 多尺度对齐：将三层特征对齐到中间尺度
2. 多核空间聚焦：多核（3/5/7/9）深度可分离卷积 + 动态门控
3. Haar频率分解：分离低频（结构）和高频（细节），分别增强后融合
4. 引导式跨尺度对齐：利用discrepancy + consistency信息

适用场景：小目标检测，需要优化跨尺度特征融合的任务
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....core import register


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ADown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class _AlignedFocusInputs(nn.Module):
    def __init__(self, inc, hidc, guided=False):
        super().__init__()
        self.low_to_mid = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )
        self.mid_proj = Conv(inc[1], hidc, 1) if inc[1] != hidc else nn.Identity()
        self.high_to_mid = ADown(inc[2], hidc)
        self.guided = guided

        if guided:
            self.low_align = _CrossScaleGuidedAlign(hidc)
            self.high_align = _CrossScaleGuidedAlign(hidc)

    def forward(self, x):
        x_low, x_mid, x_high = x
        x_low = self.low_to_mid(x_low)
        x_mid = self.mid_proj(x_mid)
        x_high = self.high_to_mid(x_high)

        if self.guided:
            x_low = self.low_align(x_low, x_mid)
            x_high = self.high_align(x_high, x_mid)

        return x_low, x_mid, x_high


class _MultiKernelSpatialFocus(nn.Module):
    def __init__(self, channels, kernel_sizes, dynamic=False):
        super().__init__()
        self.dynamic = dynamic
        self.dw_conv = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=k, padding=autopad(k), groups=channels)
            for k in kernel_sizes
        )
        self.pw_conv = Conv(channels, channels, 1)

        if dynamic:
            hidden = max(channels // 4, len(kernel_sizes) + 1)
            self.kernel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, len(kernel_sizes) + 1, kernel_size=1, bias=True),
            )

    def forward(self, x):
        branches = [x] + [layer(x) for layer in self.dw_conv]
        if self.dynamic:
            branch_weights = torch.softmax(self.kernel_gate(x), dim=1).unsqueeze(2)
            stacked = torch.stack(branches, dim=1)
            feature = torch.sum(branch_weights * stacked, dim=1)
        else:
            feature = torch.sum(torch.stack(branches, dim=0), dim=0)
        return self.pw_conv(feature)


class _HaarFrequencyDecomposition(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        weights = torch.ones(4, 1, 2, 2)
        weights[1, 0, 0, 1] = -1
        weights[1, 0, 1, 1] = -1
        weights[2, 0, 1, 0] = -1
        weights[2, 0, 1, 1] = -1
        weights[3, 0, 1, 0] = -1
        weights[3, 0, 0, 1] = -1
        self.register_buffer('weights', torch.cat([weights] * channels, dim=0), persistent=False)

    def forward(self, x):
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        out = F.conv2d(x, self.weights, bias=None, stride=2, groups=self.channels) / 4.0
        batch_size, _, height, width = out.shape
        out = out.view(batch_size, self.channels, 4, height, width)
        low = out[:, :, 0]
        high = out[:, :, 1:].abs().sum(dim=2)
        return low, high


class _CrossScaleGuidedAlign(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.context = Conv(channels * 4, channels, 1)
        self.refine = Conv(channels, channels, 3, g=channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, source, target):
        discrepancy = torch.abs(source - target)
        consistency = source * target
        context = self.context(torch.cat([source, target, discrepancy, consistency], dim=1))
        return source + self.gate(context) * self.refine(context)


class FocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
        super().__init__()
        hidc = int(inc[1] * e)

        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()
        self.conv3 = ADown(inc[2], hidc)

        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
        self.pw_conv = Conv(hidc * 3, hidc * 3)
        self.conv_1x1 = Conv(hidc * 3, int(hidc / e))

    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        feature = self.pw_conv(feature)

        x = x + feature
        return self.conv_1x1(x)


class DynamicFrequencyFocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5):
        super().__init__()
        hidc = int(inc[1] * e)
        channels = hidc * 3

        self.align = _AlignedFocusInputs(inc, hidc, guided=False)
        self.spatial_focus = _MultiKernelSpatialFocus(channels, kernel_sizes, dynamic=True)
        self.frequency = _HaarFrequencyDecomposition(channels)
        self.low_proj = Conv(channels, channels, 1)
        self.high_proj = Conv(channels, channels, 1)
        self.freq_proj = Conv(channels * 2, channels, 3)
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, max(channels // 4, 8), kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), 2, kernel_size=1, bias=True),
        )
        self.spatial_scale = nn.Parameter(torch.tensor(1.0))
        self.frequency_scale = nn.Parameter(torch.tensor(0.1))
        self.output = Conv(channels, int(hidc / e), 1)

    def forward(self, x):
        x_low, x_mid, x_high = self.align(x)
        fused = torch.cat([x_low, x_mid, x_high], dim=1)

        spatial_feature = self.spatial_focus(fused)

        low, high = self.frequency(fused)
        frequency_feature = self.freq_proj(torch.cat([self.low_proj(low), self.high_proj(high)], dim=1))
        frequency_feature = F.interpolate(
            frequency_feature,
            size=fused.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

        branch_logits = self.branch_gate(torch.cat([spatial_feature, frequency_feature], dim=1))
        branch_weights = torch.softmax(branch_logits, dim=1)
        spatial_weight, frequency_weight = torch.chunk(branch_weights, 2, dim=1)

        refined = (
            fused
            + self.spatial_scale * spatial_weight * spatial_feature
            + self.frequency_scale * frequency_weight * frequency_feature
        )
        return self.output(refined)


class AlignmentGuidedFocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5):
        super().__init__()
        hidc = int(inc[1] * e)

        self.align = _AlignedFocusInputs(inc, hidc, guided=True)
        self.discrepancy_proj = Conv(hidc * 3, hidc, 3)
        self.consistency_proj = Conv(hidc * 3, hidc, 3)
        self.branch_gate = nn.Sequential(
            Conv(hidc * 5, hidc, 1),
            nn.Conv2d(hidc, 3, kernel_size=1, bias=True),
        )
        self.guidance_residual = nn.Sequential(
            Conv(hidc * 2, hidc, 1),
            Conv(hidc, hidc, 3, g=hidc),
        )
        self.refine_focus = _MultiKernelSpatialFocus(hidc, kernel_sizes, dynamic=True)
        self.guidance_scale = nn.Parameter(torch.tensor(0.1))
        self.output = Conv(hidc, int(hidc / e), 1)

    def forward(self, x):
        x_low, x_mid, x_high = self.align(x)

        discrepancy = self.discrepancy_proj(
            torch.cat(
                [
                    torch.abs(x_low - x_mid),
                    torch.abs(x_mid - x_high),
                    torch.abs(x_low - x_high),
                ],
                dim=1,
            )
        )
        consistency = self.consistency_proj(
            torch.cat(
                [
                    x_low * x_mid,
                    x_mid * x_high,
                    x_low * x_high,
                ],
                dim=1,
            )
        )

        branch_logits = self.branch_gate(torch.cat([x_low, x_mid, x_high, discrepancy, consistency], dim=1))
        branch_weights = torch.softmax(branch_logits, dim=1)
        low_weight, mid_weight, high_weight = torch.chunk(branch_weights, 3, dim=1)

        fused = low_weight * x_low + mid_weight * x_mid + high_weight * x_high
        guidance = self.guidance_residual(torch.cat([discrepancy, consistency], dim=1))
        refined = fused + self.refine_focus(fused) + self.guidance_scale * guidance
        return self.output(refined)


@register()
class FDPN(nn.Module):
    """
    FocusingDiffusionPyramidNetwork (FDPN)
    
    聚焦扩散特征金字塔网络，用于替换HybridEncoder的FPN+PAN结构。
    
    Args:
        in_channels (list): 输入特征图的通道数列表
        feat_strides (list): 输入特征图的步幅列表
        hidden_dim (int): 隐藏层维度
        nhead (int): Transformer编码器中多头自注意力的头数
        dim_feedforward (int): Transformer编码器中前馈网络的维度
        dropout (float): Transformer编码器中的dropout概率
        enc_act (str): Transformer编码器中的激活函数类型
        use_encoder_idx (list): 指定哪些层使用Transformer编码器
        num_encoder_layers (int): Transformer编码器的层数
        pe_temperature (int): 位置编码的温度参数
        fdpn_ks (list): FDPN中的FocusFeature-kernel_sizes参数
        depth_mult (float): 深度乘数
        out_strides (list): 输出特征图的步幅列表
        eval_spatial_size (list): 评估时的空间尺寸 [H, W]
    """

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 fdpn_ks=[3, 5, 7, 9],
                 depth_mult=1.0,
                 out_strides=[8, 16, 32],
                 eval_spatial_size=None,
                 ):
        super().__init__()
        from ....deim.hybrid_encoder import TransformerEncoderLayer, TransformerEncoder

        # 保存传入的参数为类的成员变量
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = out_strides

        assert len(in_channels) == 3

        # 输入投影层
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm2d(hidden_dim))
            ]))
            self.input_proj.append(proj)

        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
        )
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # 第一阶段
        self.FocusFeature_1 = FocusFeature(inc=[hidden_dim, hidden_dim, hidden_dim], kernel_sizes=fdpn_ks)

        self.p4_to_p5_down1 = Conv(hidden_dim, hidden_dim, k=3, s=2)
        self.p5_block1 = C2f(hidden_dim * 2, hidden_dim, round(3 * depth_mult), shortcut=True)

        self.p4_to_p3_up1 = nn.Upsample(scale_factor=2)
        self.p3_block1 = C2f(hidden_dim * 2, hidden_dim, round(3 * depth_mult), shortcut=True)

        # 第二阶段
        self.FocusFeature_2 = FocusFeature(inc=[hidden_dim, hidden_dim, hidden_dim], kernel_sizes=fdpn_ks)

        self.p4_to_p5_down2 = Conv(hidden_dim, hidden_dim, k=3, s=2)
        self.p5_block2 = C2f(hidden_dim * 3, hidden_dim, round(3 * depth_mult), shortcut=True)

        if len(out_strides) == 3:
            self.p4_to_p3_up2 = nn.Upsample(scale_factor=2)
            self.p3_block2 = C2f(hidden_dim * 3, hidden_dim, round(3 * depth_mult), shortcut=True)

        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature
                )
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)

        # 输入投影
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # Transformer编码器
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        fouce_feature1 = self.FocusFeature_1(proj_feats[::-1])

        fouce_feature1_to_p5_1 = self.p4_to_p5_down1(fouce_feature1)
        fouce_feature1_to_p5_2 = self.p5_block1(torch.cat([fouce_feature1_to_p5_1, proj_feats[2]], dim=1))

        fouce_feature1_to_p3_1 = self.p4_to_p3_up1(fouce_feature1)
        fouce_feature1_to_p3_2 = self.p3_block1(torch.cat([fouce_feature1_to_p3_1, proj_feats[0]], dim=1))

        fouce_feature2 = self.FocusFeature_2([fouce_feature1_to_p5_2, fouce_feature1, fouce_feature1_to_p3_2])

        fouce_feature2_to_p5 = self.p4_to_p5_down2(fouce_feature2)
        fouce_feature2_to_p5 = self.p5_block2(torch.cat([fouce_feature2_to_p5, fouce_feature1_to_p5_1, fouce_feature1_to_p5_2], dim=1))

        if len(self.out_strides) == 3:
            fouce_feature2_to_p3 = self.p4_to_p3_up2(fouce_feature2)
            fouce_feature2_to_p3 = self.p3_block2(torch.cat([fouce_feature2_to_p3, fouce_feature1_to_p3_1, fouce_feature1_to_p3_2], dim=1))
            return [fouce_feature2_to_p3, fouce_feature2, fouce_feature2_to_p5]
        else:
            return [fouce_feature2, fouce_feature2_to_p5]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs, image_height, image_width = 1, 640, 640
    params = {
        'in_channels': [512, 1024, 2048],
        'feat_strides': [8, 16, 32],
        'hidden_dim': 128,
        'use_encoder_idx': [2],
        'fdpn_ks': [3, 5, 7, 9],
        'depth_mult': 1.0,
        'out_strides': [16, 32],
        'eval_spatial_size': [image_height, image_width]
    }

    feats = [torch.randn((bs, params['in_channels'][i], image_height // params['feat_strides'][i], image_width // params['feat_strides'][i])).to(device) for i in range(len(params['in_channels']))]
    module = FDPN(**params).to(device)
    outputs = module(feats)

    input_feats_info = ', '.join([str(i.size()) for i in feats])
    print(f'input feature:[{input_feats_info}]')
    output_feats_info = ', '.join([str(i.size()) for i in outputs])
    print(f'output feature:[{output_feats_info}]')
