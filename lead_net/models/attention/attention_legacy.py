"""LCA (Lightweight Coordinate-aware Attention) 注意力模块。

设计决策（面向 OpenMV H7 Plus 的超轻量避障优化）：
    1. Edge-aware LCA (特征梯度边缘引导)：直接计算特征图相邻像素的差值绝对值（dx, dy），
       通过无参数的梯度图表征边缘，不引入额外卷积层。
    2. Adaptive Reduction Ratio (自适应通道压缩)：压缩比根据通道数动态自适应，
       设定为 max(8, channels // 16)，消除手动配置瓶颈。
    3. Residual Attention Gate (残差门控)：添加可学习标量参数 alpha（初始为 0.0），
       输出形式为 Feature + alpha * Attention * Feature，保证训练初期梯度平稳。
    4. Obstacle Prior Mask (固定空间先验)：在注意力图上叠加热点偏置，强化中下方（避障重点区域）的权重。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCA(nn.Module):
    """Redesigned Lightweight Coordinate-aware Attention for Obstacle Detection.

    Args:
        channels: 输入/输出通道数 C。
        reduction: 通道缩减比 r，若启用自适应则会被动态覆盖。
        edge_guidance: 是否启用特征梯度边缘引导。
        obstacle_prior: 是否启用固定空间先验。
        residual_gate: 是否启用残差门控。
    """

    def __init__(self, channels: int, reduction: int = 16,
                 edge_guidance: bool = True, obstacle_prior: bool = True,
                 residual_gate: bool = True):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.edge_guidance = edge_guidance
        self.obstacle_prior = obstacle_prior
        self.residual_gate = residual_gate

        mip = max(8, channels // reduction)
        self.mip = mip

        # 方向 1D 池化（自适应：保留指定方向的完整尺寸）
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 沿 W 池化 → [B,C,H,1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 沿 H 池化 → [B,C,1,W]

        # concat 后通道缩减 + 归一化 + 非线性
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        # 各方向 1x1 conv 还原通道数
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

        # 残差门控的可学习缩放因子
        if self.residual_gate:
            self.alpha = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in (self.conv1, self.conv_h, self.conv_w):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        # 1. Edge-aware LCA (特征梯度边缘引导)
        if self.edge_guidance:
            # 计算水平和垂直方向特征梯度差绝对值
            dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            dx = F.pad(dx, (1, 0, 0, 0))  # 填充左侧保持 W 维度
            dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
            dy = F.pad(dy, (0, 0, 1, 0))  # 填充上方保持 H 维度
            edge_map = dx + dy
            g_e = torch.sigmoid(edge_map)

        # 2. Coordinate Attention 位置编码
        x_h = self.pool_h(x)                  # [B,C,H,1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B,C,W,1]

        # concat 方向编码 (沿 H 方向拼接成 [B, C, H+W, 1])
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # split 回两个方向
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)         # 回到 [B,C,1,W]

        # 方向门控权重
        g_h = torch.sigmoid(self.conv_h(x_h))  # [B,C,H,1]
        g_w = torch.sigmoid(self.conv_w(x_w))  # [B,C,1,W]

        # 门控图合并
        att = g_h * g_w  # [B,C,H,W]

        # 3. Obstacle-aware Prior (固定空间先验)
        if self.obstacle_prior:
            # 生成中下方高亮（Y=0.75, X=0.5）的 2D 空间先验 Mask
            y_coords = torch.linspace(0, 1, steps=h, dtype=x.dtype, device=x.device)
            x_coords = torch.linspace(0, 1, steps=w, dtype=x.dtype, device=x.device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            
            sigma_y = 0.40
            sigma_x = 0.35
            dist_y = (grid_y - 0.75) ** 2 / (2 * (sigma_y ** 2))
            dist_x = (grid_x - 0.5) ** 2 / (2 * (sigma_x ** 2))
            prior = torch.exp(-(dist_y + dist_x))
            prior = 0.5 + 0.7 * prior  # 归一化映射到 [0.5, 1.2]
            prior = prior.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            att = att * prior

        # 融合边缘引导
        if self.edge_guidance:
            att = att * g_e

        # 4. Residual Attention Gate (残差门控)
        if self.residual_gate:
            out = identity + self.alpha * att * identity
        else:
            out = identity * att

        return out


def build_lca(cfg: dict, channels: int) -> LCA:
    """按 cfg.model.lca 构造 LCA 模块。

    Args:
        cfg: 完整配置；读取 cfg["model"]["lca"]。
        channels: LCA 所在特征图的通道数。
    """
    lca_cfg: dict = cfg.get("model", {}).get("lca", {})
    
    edge_guidance = lca_cfg.get("edge_guidance", True)
    obstacle_prior = lca_cfg.get("obstacle_prior", True)
    residual_gate = lca_cfg.get("residual_gate", True)
    
    # 自适应压缩率
    adaptive_reduction = lca_cfg.get("adaptive_reduction", True)
    if adaptive_reduction:
        reduction = max(8, channels // 16)
    else:
        reduction = int(lca_cfg.get("reduction", 16))

    return LCA(
        channels=channels,
        reduction=reduction,
        edge_guidance=edge_guidance,
        obstacle_prior=obstacle_prior,
        residual_gate=residual_gate
    )