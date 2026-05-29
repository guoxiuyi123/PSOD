"""
自研模块：Dynamic Wavelet-Gated Feature Sampling (WGFS)
基于动态频域门控的小波下采样网络

核心创新：
1. Haar小波频段分解：将输入分解为LL/LH/HL/HH四个正交子带
2. 动态频段门控：通过通道注意力机制自适应计算频段重要性
3. 空频域双支路融合：结合空间域卷积和频域小波特征

适用场景：小目标检测，需要保留高频细节（边缘、纹理）的任务
"""

import torch
import torch.nn as nn


class DynamicHaarCut(nn.Module):
    """
    Dynamic Haar Wavelet Gating (动态小波门控)
    
    不仅进行正交频域分解，还利用通道注意力机制，
    根据输入图像动态计算4个频段的重要性权重。
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 动态瓶颈层：根据输入通道数调整
        bottleneck = max(in_channels // 4, 16)
        
        # 轻量级的"频段注意力生成器"
        self.frequency_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # 提取全局空间上下文
            nn.Conv2d(in_channels * 4, bottleneck, 1),  # 降维瓶颈层
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, 4, 1)       # 预测LL, LH, HL, HH四个频段的动态权重
        )
        
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 2x2子像素分解
        x00 = x[:, :, 0::2, 0::2]  # top-left
        x01 = x[:, :, 0::2, 1::2]  # top-right
        x10 = x[:, :, 1::2, 0::2]  # bottom-left
        x11 = x[:, :, 1::2, 1::2]  # bottom-right

        # 2D Haar小波频段分解
        LL = (x00 + x01 + x10 + x11) * 0.5   # 低频结构
        LH = (x00 - x01 + x10 - x11) * 0.5   # 水平高频细节
        HL = (x00 + x01 - x10 - x11) * 0.5   # 垂直高频细节
        HH = (x00 - x01 - x10 + x11) * 0.5   # 对角线高频细节

        # 将四个频段拼接用于计算动态注意力
        x_concat = torch.cat([LL, LH, HL, HH], dim=1)
        
        # [B, 4, 1, 1] - 针对每张图片的自适应频段重要性得分
        dynamic_bw = torch.sigmoid(self.frequency_attention(x_concat))
        
        # 将动态权重分别应用到四个物理频段上
        w_LL = dynamic_bw[:, 0:1, :, :]
        w_LH = dynamic_bw[:, 1:2, :, :]
        w_HL = dynamic_bw[:, 2:3, :, :]
        w_HH = dynamic_bw[:, 3:4, :, :]
        
        x_gated = torch.cat([LL * w_LL, LH * w_LH, HL * w_HL, HH * w_HH], dim=1)
        
        return self.batch_norm(self.conv_fusion(x_gated))


class WGFS(nn.Module):
    """
    基于动态频域门控的小波下采样网络 (Dynamic Wavelet-Gated Feature Sampling)
    
    两阶段下采样：
    - Stage 1: 原始尺寸 → 2x下采样（空频域双支路融合）
    - Stage 2: 2x → 4x下采样（卷积、池化、小波三支路融合）
    
    Args:
        in_channels (int): 输入通道数，默认3（RGB图像）
        out_channels (int): 输出通道数
    """
    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        out_c14 = max(int(out_channels / 4), 4)
        out_c12 = max(int(out_channels / 2), 8)

        self.conv_init = nn.Conv2d(in_channels, out_c14, kernel_size=7, stride=1, padding=3)

        # Stage 1: original → 2x downsampling
        self.conv_1 = nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14)
        self.conv_x1 = nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12)
        self.batch_norm_x1 = nn.BatchNorm2d(out_c12)
        self.haar_cut_c = DynamicHaarCut(out_c14, out_c12)
        self.fusion1 = nn.Conv2d(out_channels, out_c12, kernel_size=1, stride=1)

        # Stage 2: 2x → 4x downsampling
        self.conv_2 = nn.Conv2d(out_c12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_c12)
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.haar_cut_r = DynamicHaarCut(out_c12, out_channels)
        self.fusion2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv_init(x)  # [B, C/4, H, W]

        # Stage 1 (空频域双支路融合)
        c = self.haar_cut_c(x)
        xc = self.batch_norm_x1(self.conv_x1(self.conv_1(x)))
        x = self.fusion1(torch.cat([xc, c], dim=1))  # [B, C/2, H/2, W/2]

        # Stage 2 (卷积、池化、小波三支路融合)
        r = x
        x_exp = self.conv_2(r)
        conv_d = self.batch_norm_x2(self.conv_x2(x_exp))
        max_d = self.batch_norm_m(self.max_m(x_exp))
        haar_d = self.haar_cut_r(r)
        x = self.fusion2(torch.cat([conv_d, haar_d, max_d], dim=1))  # [B, C, H/4, W/4]
        return x


if __name__ == '__main__':
    # 测试模块
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试WGFS
    wgfs = WGFS(in_channels=3, out_channels=16).to(device)
    x = torch.randn(1, 3, 640, 640).to(device)
    out = wgfs(x)
    print(f"WGFS: input {x.shape} -> output {out.shape}")
