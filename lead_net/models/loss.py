"""MultiBox Loss (SSD 损失函数)。

依据 SSD 论文 arXiv:1512.02325：
    - 分类损失：CrossEntropy（含背景类）
    - 回归损失：Smooth L1（仅正样本）
    - 锚框匹配：IoU > threshold 为正样本
    - 难负例挖掘：负:正 = 3:1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    """SSD MultiBox 损失。

    Args:
        num_classes: 含背景的类别数
        overlap_threshold: 正样本匹配 IoU 阈值
        neg_pos_ratio: 负样本/正样本最大比例
        variance: 偏移编码 variance (cx,cy 用 variances[0], w,h 用 variances[1])
    """

    def __init__(self, num_classes: int, input_size: int = 320,
                 overlap_threshold: float = 0.5, neg_pos_ratio: int = 3,
                 variance: tuple[float, float] = (0.1, 0.2)):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.overlap_threshold = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.variance = variance

    def forward(self, cls_pred: torch.Tensor, loc_pred: torch.Tensor,
                default_boxes: torch.Tensor,
                gt_boxes: list[torch.Tensor],
                gt_labels: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """计算损失。

        Args:
            cls_pred: [B, N, num_classes] 分类预测
            loc_pred: [B, N, 4] 回归预测
            default_boxes: [N, 4] 预定义锚框 (cx,cy,w,h normalized [0,1])
            gt_boxes: list[Tensor[n_i, 4]], 每张图的 GT boxes (COCO xywh, 绝对像素)
            gt_labels: list[Tensor[n_i]], 每张图的 GT labels (不含背景，0-indexed)

        Returns:
            cls_loss, loc_loss
        """
        device = cls_pred.device
        B = cls_pred.size(0)

        if default_boxes.device != device:
            default_boxes = default_boxes.to(device)

        loc_t = []    # 编码后的 GT offsets
        conf_t = []   # GT class labels (0=background)
        for b in range(B):
            gtb = gt_boxes[b].to(device)
            gtl = gt_labels[b].to(device)
            if len(gtb) == 0:
                loc_t.append(torch.zeros((cls_pred.size(1), 4), device=device))
                conf_t.append(torch.zeros(cls_pred.size(1), dtype=torch.long, device=device))
                continue

            # Normalize GT: 绝对像素 xywh → 归一化 [0,1] xywh → xyxy
            gtb_norm = gtb.float() / self.input_size
            gtb_xyxy = _xywh_to_xyxy(gtb_norm)

            matches = _match(default_boxes, gtb_xyxy, self.overlap_threshold)
            pos_mask = matches >= 0

            loc_target = torch.zeros((cls_pred.size(1), 4), device=device)
            conf_target = torch.zeros(cls_pred.size(1), dtype=torch.long, device=device)

            if pos_mask.any():
                pos_indices = torch.where(pos_mask)[0]
                matched_gts = matches[pos_indices]
                loc_target[pos_indices] = _encode(
                    default_boxes[pos_indices],
                    gtb_xyxy[matched_gts],
                    self.variance,
                )
                conf_target[pos_indices] = gtl[matched_gts] + 1  # +1 for background

            loc_t.append(loc_target)
            conf_t.append(conf_target)

        loc_t = torch.stack(loc_t, dim=0)
        conf_t = torch.stack(conf_t, dim=0)

        # 正样本 mask
        pos = conf_t > 0  # [B, N]
        num_pos = pos.sum(dim=1)  # [B]

        # 定位损失 (仅正样本)
        loc_loss = F.smooth_l1_loss(
            loc_pred[pos], loc_t[pos], reduction="sum"
        )

        # 分类损失
        cls_loss = _hard_negative_mining(
            cls_pred, conf_t, pos, num_pos, self.neg_pos_ratio
        )

        # Normalize by num_pos（数值稳定性：min=10 防止单样本梯度爆炸）
        # 参考：SSD 论文 + YOLO 社区实践，过小的分母导致 loss 量级异常
        total_pos = num_pos.sum().float().clamp(min=10)
        cls = cls_loss / total_pos
        loc = loc_loss / total_pos

        # NaN/Inf 安全检查（训练初期可能因标注问题触发）
        if torch.isnan(cls) or torch.isinf(cls):
            cls = torch.tensor(0.0, device=device, requires_grad=True)
        if torch.isnan(loc) or torch.isinf(loc):
            loc = torch.tensor(0.0, device=device, requires_grad=True)

        return cls, loc


def _match(default_boxes: torch.Tensor, gt_boxes: torch.Tensor,
           threshold: float) -> torch.Tensor:
    """为每个 default box 匹配 GT box。

    Returns:
        matches: [num_defaults], -1 表示负样本，否则为 GT index
    """
    from torchvision.ops import box_iou

    # default_boxes in cxcywh → xyxy; gt_boxes already in xyxy
    d_xyxy = _cxcywh_to_xyxy(default_boxes)   # [N, 4]
    g_xyxy = gt_boxes                          # [M, 4] already xyxy

    ious = box_iou(d_xyxy, g_xyxy)  # [N, M]

    # 每个 GT 匹配到 best default
    best_d_per_g = ious.max(dim=0).indices  # [M]
    # 每个 default 匹配到 best GT
    best_g_per_d = ious.max(dim=1)
    best_g_idx = best_g_per_d.indices  # [N]
    best_g_iou = best_g_per_d.values   # [N]

    # 确保每个 GT 至少匹配到一个 default
    best_g_idx[best_d_per_g] = torch.arange(len(gt_boxes), device=ious.device)

    # 低于阈值的为负样本
    best_g_idx[best_g_iou < threshold] = -1

    return best_g_idx


def _encode(default_boxes: torch.Tensor, matched_gt: torch.Tensor,
            variance: tuple[float, float]) -> torch.Tensor:
    """将 GT boxes 编码为相对 default boxes 的偏移量。"""
    g_cx = (matched_gt[:, 0] + matched_gt[:, 2]) / 2
    g_cy = (matched_gt[:, 1] + matched_gt[:, 3]) / 2
    g_w = matched_gt[:, 2] - matched_gt[:, 0]
    g_h = matched_gt[:, 3] - matched_gt[:, 1]

    d_cx, d_cy, d_w, d_h = default_boxes.unbind(-1)

    loc_cx = (g_cx - d_cx) / (d_w * variance[0])
    loc_cy = (g_cy - d_cy) / (d_h * variance[0])
    eps = 1e-6
    loc_w = torch.log((g_w + eps) / (d_w + eps)) / variance[1]
    loc_h = torch.log((g_h + eps) / (d_h + eps)) / variance[1]

    return torch.stack([loc_cx, loc_cy, loc_w, loc_h], dim=-1)


def _hard_negative_mining(cls_pred: torch.Tensor, conf_t: torch.Tensor,
                          pos: torch.Tensor, num_pos: torch.Tensor,
                          neg_pos_ratio: int) -> torch.Tensor:
    """Hard negative mining 后的分类 CrossEntropy 损失。"""
    import math
    B = cls_pred.size(0)
    cls_loss_all = F.cross_entropy(
        cls_pred.view(-1, cls_pred.size(-1)),
        conf_t.view(-1),
        reduction="none",
    ).view(B, -1)

    loss = torch.zeros(1, device=cls_pred.device)
    for b in range(B):
        pos_loss = cls_loss_all[b][pos[b]].sum() if pos[b].any() else 0.0

        neg = ~pos[b]
        n_neg = min(
            neg.sum().item(),
            max(1, int(num_pos[b].item()) * neg_pos_ratio),
        )
        if n_neg > 0 and neg.any():
            neg_loss_all = cls_loss_all[b][neg]
            if neg_loss_all.numel() > 0:
                neg_loss = neg_loss_all.topk(n_neg).values.sum()
            else:
                neg_loss = 0.0
        else:
            neg_loss = 0.0

        loss = loss + pos_loss + neg_loss

    return loss


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """cx,cy,w,h → x1,y1,x2,y2 (all normalized)。"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([
        cx - w / 2, cy - h / 2,
        cx + w / 2, cy + h / 2,
    ], dim=-1)


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """x,y,w,h → x1,y1,x2,y2 (all normalized)。"""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)
