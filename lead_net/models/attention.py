"""LCA (Lightweight Coordinate-aware Attention) 注意力模块。

依据：
    - docs/PROJECT_CONTEXT.md §核心技术路线：LCA 插入于 Backbone 最后阶段
    - docs/MODULES.md §2 Attention
    - 参考论文：Coordinate Attention, CVPR 2021 (arXiv:2103.02907)

设计决策（M2 实现，理由记录于 docs/CHANGELOG.md）：
    - 直接采用 Coordinate Attention 原始结构（1D 方向池化 + concat + 1x1 通道缩减 +
      BN + Hardswish + split + 方向门控），不引入额外大核 depthwise 卷积，
      以最小参数/计算增量满足 RQ2（开销代价）约束。
    - 激活函数用 nn.Hardswish，与 MobileNetV3-Small Backbone 内部激活一致，
      保证 INT8 量化（RQ3）时算子图统一、便于 tflite 转换。
    - 通道缩减比 reduction：M2 取 16（configs/default.yaml 占位值），缩减后通道
      mip = max(8, C // reduction) 限制下界避免极端低通道导致表达力崩塌（CA 原始实现惯例）。
    - 输入/输出 shape 保持一致 [B, C, H, W]，不改变特征图尺寸与通道数，
      满足 ARCHITECTURE.md §模块接口约定（LCA 输入/输出 shape 保持一致）。
    - 插入位置：Backbone 最后阶段的最小语义特征图（stride 32 投影后 s32, [B,256,10,10]），
      10x10 特征图上方向注意力计算开销最低，符合"特征图尺寸小、计算开销最低"原则。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LCA(nn.Module):
    """Lightweight Coordinate-aware Attention。

    基于方向感知的注意力：分别沿 H、W 方向做全局池化，得到方向编码特征，
    拼接后通道缩减 + 非线性，再分裂回两个方向并经 1x1 conv + sigmoid 生成
    方向门控权重，与原特征相乘。保留精确空间位置信息（区别于 SENet 的全局标量权重）。

    Args:
        channels: 输入/输出通道数 C（LCA 不改变通道数与空间尺寸）。
        reduction: 通道缩减比 r，中间通道 mip = max(8, C // r)。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mip = max(8, channels // reduction)
        self.mip = mip

        # 方向 1D 池化（自适应：保留指定方向的完整尺寸）
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 沿 W 池化 → [B,C,H,1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 沿 H 池化 → [B,C,1,W]

        # concat 后通道缩减 + 归一化 + 非线性
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        # 各方向 1x1 conv 还原通道数 + sigmoid 门控
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

        self._init_weights()

    def _init_weights(self):
        for m in (self.conv1, self.conv_h, self.conv_w):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        # 方向池化
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

        # 广播相乘（H 方向权重沿 W 广播、W 方向权重沿 H 广播）
        out = identity * g_h * g_w
        return out


def build_lca(cfg: dict, channels: int) -> LCA:
    """按 cfg.model.lca 构造 LCA 模块。

    Args:
        cfg: 完整配置；读取 cfg["model"]["lca"]["reduction"]。
        channels: LCA 所在特征图的通道数。
    """
    lca_cfg: dict = cfg.get("model", {}).get("lca", {})
    reduction = int(lca_cfg.get("reduction", 16))
    return LCA(channels=channels, reduction=reduction)