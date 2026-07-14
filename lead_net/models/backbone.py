"""MobileNetV3-Small Backbone with multi-scale output。

依据：
    - docs/PROJECT_CONTEXT.md §核心技术路线：MobileNetV3-Small Backbone
    - docs/MODULES.md §1 Backbone：多尺度输出供 SSD-Lite 使用
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from .attention import build_lca


def build_backbone(cfg: dict) -> nn.Module:
    bk_cfg: dict[str, Any] = cfg.get("model", {}).get("backbone", {})
    name = bk_cfg.get("name", "mobilenet_v3_small")
    if name != "mobilenet_v3_small":
        raise ValueError(f"仅支持 mobilenet_v3_small, got {name!r}")

    weights_name = bk_cfg.get("weights", "IMAGENET1K_V1")
    if weights_name == "IMAGENET1K_V1":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    elif weights_name is None:
        weights = None
    else:
        raise ValueError(f"未支持的 weights: {weights_name!r}")

    width_multiplier = bk_cfg.get("width_multiplier", 1.0)
    if width_multiplier != 1.0:
        raise NotImplementedError("width_multiplier != 1.0 暂不支持")

    use_lca: bool = cfg.get("model", {}).get("lca", {}).get("enabled", False)
    full = mobilenet_v3_small(weights=weights)
    return MobileNetV3SmallBackbone(
        features=full.features,
        input_size=cfg.get("data", {}).get("input_size", 320),
        use_lca=use_lca,
        lca_cfg=cfg,
    )


class MobileNetV3SmallBackbone(nn.Module):
    """MobileNetV3-Small 多尺度特征提取。

    输出 3 个 scale 的特征图：
        - stride 16: 20x20, 48ch → 投影到 256ch
        - stride 32: 10x10, 576ch → 投影到 256ch
        - stride 64: 5x5, 128ch (额外卷积层)

    LCA（M2）：可选插入到 stride 32 投影后的 s32 [B,256,10,10]（backbone 最后阶段
        最小语义特征图），不改变 out_channels 与 fm_sizes。
    """

    def __init__(self, features: nn.Sequential, input_size: int = 320,
                 use_lca: bool = False, lca_cfg: dict | None = None):
        super().__init__()
        self.features = features
        self.input_size = input_size
        self.use_lca = use_lca

        # 提取点：features[8] 输出 stride 16, 48ch; features[12] 输出 stride 32, 576ch
        # 额外层从 features[12] 继续下采样
        self.extra = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 通道投影：将 stride 16 的 48ch 投影到 256ch
        self.proj_s16 = nn.Sequential(
            nn.Conv2d(48, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 通道投影：将 stride 32 的 576ch 投影到 256ch
        self.proj_s32 = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self._out_channels = [256, 256, 128]
        self._fm_sizes = [input_size // 16, input_size // 32, input_size // 64]

        # LCA 注入到 stride-32 投影后特征（backbone 最后阶段最小语义特征图）
        self.lca: nn.Module | None = None
        if use_lca:
            if lca_cfg is None:
                lca_cfg = {}
            self.lca = build_lca(lca_cfg, channels=256)

        self._init_extra()

    def _init_extra(self):
        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.proj_s16.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for m in self.proj_s32.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """返回多尺度特征图 [s16, s32, s64]."""
        # 逐层前向直到 features[8]
        feat = x
        f_s16 = None
        for i in range(8 + 1):
            feat = self.features[i](feat)
            if i == 8:
                f_s16 = feat  # [B, 48, 20, 20]

        # 继续到 features[12]
        for i in range(9, 13):
            feat = self.features[i](feat)
        f_s32 = feat  # [B, 576, 10, 10]

        # 额外层
        f_s64 = self.extra(f_s32)  # [B, 128, 5, 5]

        # 通道投影
        s16 = self.proj_s16(f_s16)
        s32 = self.proj_s32(f_s32)
        if self.lca is not None:
            s32 = self.lca(s32)

        return [s16, s32, f_s64]

    @property
    def out_channels(self) -> list[int]:
        return self._out_channels

    @property
    def fm_sizes(self) -> list[int]:
        return self._fm_sizes
