"""LCA (Lightweight Coordinate-aware Attention) — ultralytics 适配版。

基于 lead_net/models/attention.py 的 LCA 设计，适配 ultralytics YAML 模块接口：
    - forward(x) 单输入单输出（ultralytics 模块约定）
    - args 解析：(channels, reduction) 或 (channels,) + cfg
    - 训练版/部署版双实现（deploy=True 时用 TFLite 友好算子）

设计决策（面向 OpenMV H7 Plus 超轻量避障优化）：
    1. Edge-aware LCA: 特征梯度边缘引导
    2. Adaptive Reduction: 压缩比 max(8, channels // reduction)
    3. Residual Attention Gate: 可学习 alpha，训练初期梯度平稳
    4. Obstacle Prior: 中下方空间先验（部署版预计算为常量）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCA(nn.Module):
    """Lightweight Coordinate-aware Attention for Obstacle Detection.

    Args:
        channels: 输入/输出通道数 C
        reduction: 通道缩减比 r（实际 mip = max(8, channels // reduction)）
        edge_guidance: 特征梯度边缘引导
        obstacle_prior: 固定空间先验（中下方高亮）
        residual_gate: 残差门控（可学习 alpha）
        deploy: 部署模式（TFLite 友好算子，预计算 prior）
    """

    def __init__(self, channels: int | None = None, reduction: int = 16,
                 edge_guidance: bool = True, obstacle_prior: bool = True,
                 residual_gate: bool = True, deploy: bool = False):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.edge_guidance = edge_guidance
        self.obstacle_prior = obstacle_prior
        self.residual_gate = residual_gate
        self.deploy = deploy

        # channels 可为 None（YAML 只传 reduction，parse_model 不注入 c1）
        # 此时构造不建层，首次 forward 按 x.size(1) lazy 重建
        self._real_channels: int | None = None
        self.mip: int = 0
        self._init_done = False

        if channels and channels > 0:
            self._build_layers(channels)
            self._init_done = True

        if self.residual_gate:
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("alpha", None)

        self._prior_buffer: torch.Tensor | None = None

    def _build_layers(self, channels: int) -> None:
        """按真实 channels 建 conv/bn 层，BN identity 初始化保证不破坏预训练。"""
        mip = max(8, channels // max(self.reduction, 1))
        self.mip = mip
        self._real_channels = channels
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1)
        self._init_weights()
        # BN identity 初始化：running_mean=0, running_var=1, weight=1, bias=0
        # 保证初始前向输出 ≈ identity（配合 alpha=0 的 residual gate）
        for bn in [self.bn1]:
            nn.init.constant_(bn.running_mean, 0.0)
            nn.init.constant_(bn.running_var, 1.0)
            nn.init.constant_(bn.weight, 1.0)
            nn.init.constant_(bn.bias, 0.0)
        # conv_h/conv_w 输出经 sigmoid，初始化为 0 → sigmoid=0.5（中性门控）
        nn.init.zeros_(self.conv_h.weight)
        nn.init.zeros_(self.conv_w.weight)
        if self.conv_h.bias is not None:
            nn.init.zeros_(self.conv_h.bias)
        if self.conv_w.bias is not None:
            nn.init.zeros_(self.conv_w.bias)

    def _init_weights(self):
        for m in (self.conv1, self.conv_h, self.conv_w):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_prior(self, h: int, w: int, dtype, device) -> torch.Tensor:
        """中下方高亮 2D 先验，部署版预计算缓存。"""
        y_coords = torch.linspace(0, 1, steps=h, dtype=dtype, device=device)
        x_coords = torch.linspace(0, 1, steps=w, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        sigma_y, sigma_x = 0.40, 0.35
        prior = torch.exp(-((grid_y - 0.75) ** 2 / (2 * sigma_y ** 2)
                            + (grid_x - 0.5) ** 2 / (2 * sigma_x ** 2)))
        return 0.5 + 0.7 * prior  # [0.5, 1.2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        # Lazy 初始化：按真实输入 channels 建层
        if not self._init_done or self._real_channels != c:
            self._build_layers(c)
            self.to(x.device)
            self._init_done = True
            # 刚重建时，BN 用 running stats（eval 模式）避免 batch stats 扰动
            self.bn1.eval()

        # 如果 alpha 很小（<0.01），LCA 近似 identity，不计算 att 节省算力
        if self.residual_gate and abs(self.alpha.item()) < 1e-4:
            return identity

        # BN 在训练初期保持 eval（用 running stats），alpha 增大后切回 train
        if self.training and self.residual_gate and self.alpha.item() < 0.05:
            self.bn1.eval()
        elif self.training:
            self.bn1.train()

        # 1. Edge guidance（部署版用 abs 标准算子）
        if self.edge_guidance:
            dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            dx = F.pad(dx, (1, 0, 0, 0))
            dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
            dy = F.pad(dy, (0, 0, 1, 0))
            g_e = torch.sigmoid(dx + dy)
        else:
            g_e = None

        # 2. Coordinate attention
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        g_h = torch.sigmoid(self.conv_h(x_h))
        g_w = torch.sigmoid(self.conv_w(x_w))
        att = g_h * g_w

        # 3. Obstacle prior
        if self.obstacle_prior:
            if self.deploy and self._prior_buffer is not None:
                prior = self._prior_buffer.to(dtype=x.dtype, device=x.device)
                if prior.shape[-2:] != (h, w):
                    prior = F.interpolate(prior, size=(h, w), mode="bilinear", align_corners=False)
            else:
                prior = self._make_prior(h, w, x.dtype, x.device)
            att = att * prior

        if g_e is not None:
            att = att * g_e

        # 4. Residual gate
        if self.residual_gate:
            out = identity + self.alpha * att * identity
        else:
            out = identity * att
        return out

    def prepare_deploy(self) -> None:
        """切换到部署模式，预计算 prior 常量。"""
        self.deploy = True
        self._prior_buffer = self._make_prior(20, 20, torch.float32, torch.device("cpu"))


def build_lca_from_cfg(channels: int, reduction: int = 16, **kwargs) -> LCA:
    """从配置构建 LCA。"""
    return LCA(channels=channels, reduction=reduction, **kwargs)
