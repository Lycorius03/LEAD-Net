"""LEAD-Net 模型组装。

依据：
    - docs/PROJECT_CONTEXT.md §核心技术路线：
        MobileNetV3-Small Backbone → [LCA] → SSD-Lite Detection Head
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone
from .detection_head import build_detection_head


def build_lead_net(cfg: dict) -> nn.Module:
    """构建 LEAD-Net（多尺度）。

    通过 cfg.model.lca.enabled 控制是否注入 LCA 注意力（M2 实现），
    服务 RQ1（精度对比）/RQ2（开销对比）的消融实验。
    """
    use_lca: bool = cfg.get("model", {}).get("lca", {}).get("enabled", False)

    backbone = build_backbone(cfg)
    head = build_detection_head(cfg, in_channels=backbone.out_channels,
                                fm_sizes=backbone.fm_sizes)
    return LeadNet(backbone=backbone, head=head, use_lca=use_lca)


class LeadNet(nn.Module):
    """LEAD-Net 检测网络。forward 返回 (cls_pred, loc_pred)。

    use_lca 仅用于语义标记（实际 LCA 模块已挂在 backbone 内部），便于日志/消融区分。
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, use_lca: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.use_lca = use_lca

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(images)  # list of [B,C,H,W]
        return self.head(feats)
