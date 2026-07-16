"""SSD-Lite Detection Head —— 多尺度 + depthwise separable conv。

依据：
    - SSD 论文：arXiv:1512.02325；SSDLite: arXiv:1801.04381
    - docs/MODULES.md §3 Detection Head
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_detection_head(cfg: dict, in_channels: list[int] | None = None,
                         fm_sizes: list[int] | None = None) -> nn.Module:
    """构建多尺度 SSD-Lite head。"""
    num_classes: int = cfg.get("num_classes", 7) + 1  # +background
    input_size: int = cfg.get("data", {}).get("input_size", 320)

    if fm_sizes is None:
        fm_sizes = [input_size // 16, input_size // 32, input_size // 64]  # 20, 10, 5
    if in_channels is None:
        in_channels = [256, 256, 128]

    # 每尺度锚框配置：(scale, aspect_ratios)
    #
    # 锚框数量：
    #   stride 16 (20×20): 1+1+3+5 = 10/cell → 4000 anchors
    #   stride 32 (10×10): 1+5     =  6/cell →  600 anchors
    #   stride 64 ( 5×5): 5         =  5/cell →  125 anchors
    #   Total: 4725 anchors
    #
    # 设计依据（2026-07-17 数据集诊断）：
    #   - 73% 目标为小目标 (<32×32px), 最小宽度 P5=2px → 新增 scale 0.03 (≈10px)
    #   - class 6 (bottle) 仅 8.8% 自然匹配率, 宽高比 P5=0.15 → 新增 AR [3, 1/3]
    #   - 原配置仅 2,475 anchors, 每图均值 4.2 正样本 → 严重不足
    scale_configs = [
        ([0.03, 0.05, 0.1, 0.2], [[1], [1], [1, 2, 0.5], [1, 2, 0.5, 3, 1/3]]),   # stride 16: 极小→小目标
        ([0.3, 0.5],              [[1], [1, 2, 0.5, 3, 1/3]]),                      # stride 32: 中目标+极端AR
        ([0.7],                   [[1, 2, 0.5, 3, 1/3]]),                            # stride 64: 大目标+极端AR
    ]

    return SSDHead(in_channels, num_classes, input_size, fm_sizes, scale_configs)


class DefaultBoxGenerator:
    """为单个特征图生成 default boxes (cx,cy,w,h normalized)。"""

    def __init__(self, input_size: int, fm_size: int, scales: list[float],
                 aspect_ratios: list[list[int]]):
        self.input_size = input_size
        self.fm_size = fm_size
        self.stride = input_size / fm_size
        self.default_boxes = self._generate(scales, aspect_ratios)

    def _generate(self, scales: list[float],
                  aspect_ratios: list[list[int]]) -> torch.Tensor:
        boxes = []
        for y in range(self.fm_size):
            cy = (y + 0.5) * self.stride / self.input_size
            for x in range(self.fm_size):
                cx = (x + 0.5) * self.stride / self.input_size
                for scale, ars in zip(scales, aspect_ratios):
                    for ar in ars:
                        w = scale * math.sqrt(ar)
                        h = scale / math.sqrt(ar)
                        boxes.append([cx, cy, w, h])
        return torch.tensor(boxes, dtype=torch.float32)

    def to(self, device: torch.device) -> torch.Tensor:
        return self.default_boxes.to(device)


class DepthwiseSepConv(nn.Module):
    """Depthwise separable conv: depthwise 3x3 → pointwise 1x1."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self._init_weights()

    def _init_weights(self):
        for m in (self.depthwise, self.pointwise):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SSDHead(nn.Module):
    """多尺度 SSD-Lite 预测头。

    Args:
        in_channels: 各尺度输入通道数列表
        num_classes: 含背景的类别数
        input_size: 输入图像尺寸
        fm_sizes: 各尺度特征图空间尺寸列表
        scale_configs: 每尺度 (scales, aspect_ratios) 配置
    """

    def __init__(self, in_channels: list[int], num_classes: int,
                 input_size: int, fm_sizes: list[int],
                 scale_configs: list[tuple[list[float], list[list[int]]]]):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.box_generators = []  # plain list, not ModuleList
        self.cls_branches = nn.ModuleList()
        self.loc_branches = nn.ModuleList()
        self.num_anchors_per_scale = []

        for in_ch, fm_sz, (scales, ars) in zip(in_channels, fm_sizes, scale_configs):
            # 计算该尺度的锚框数
            n_anchors = sum(len(a) for a in ars)
            self.num_anchors_per_scale.append(n_anchors)

            # 锚框生成器
            gen = DefaultBoxGenerator(input_size, fm_sz, scales, ars)
            self.box_generators.append(gen)

            # Depthwise separable 分类/回归分支
            self.cls_branches.append(
                DepthwiseSepConv(in_ch, n_anchors * num_classes)
            )
            self.loc_branches.append(
                DepthwiseSepConv(in_ch, n_anchors * 4)
            )

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            features: list of [B, C_i, H_i, W_i]

        Returns:
            cls_pred: [B, total_anchors, num_classes]
            loc_pred: [B, total_anchors, 4]
        """
        cls_preds = []
        loc_preds = []

        for i, feat in enumerate(features):
            B = feat.size(0)

            cls_out = self.cls_branches[i](feat)  # [B, A*C, H, W]
            loc_out = self.loc_branches[i](feat)  # [B, A*4, H, W]

            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            cls_out = cls_out.view(B, -1, self.num_classes)
            cls_preds.append(cls_out)

            loc_out = loc_out.permute(0, 2, 3, 1).contiguous()
            loc_out = loc_out.view(B, -1, 4)
            loc_preds.append(loc_out)

        return torch.cat(cls_preds, dim=1), torch.cat(loc_preds, dim=1)

    def all_default_boxes(self, device: torch.device) -> torch.Tensor:
        """返回所有尺度的 concat 锚框 [total_anchors, 4]."""
        return torch.cat([g.to(device) for g in self.box_generators], dim=0)

    def decode(self, loc_pred: torch.Tensor, cls_pred: torch.Tensor,
               score_threshold: float = 0.5, nms_threshold: float = 0.45,
               max_detections: int = 100,
               pre_nms_topk: int = 1000,
               variance: tuple[float, float] = (0.1, 0.2)) -> list[list[dict]]:
        """解码预测为边界框并执行 NMS。

        性能优化（避免逐类 Python 循环与候选爆炸）：
            - softmax 后对 (anchor, class>=1) 统一过滤；
            - pre_nms_topk 截断每图最高分候选数（避免几万候选进 NMS）；
            - torchvision.ops.batched_nms 一次性按 class 分组 NMS（替代逐类循环）；
            - NMS 后再按 score 取 top max_detections。

        Args:
            max_detections: 每图最终保留的最大检测数。
            pre_nms_topk: 进入 NMS 前每图候选截断数（按 score 降序取前 k）。
        """
        from torchvision.ops import batched_nms

        device = loc_pred.device
        B = loc_pred.size(0)
        default_boxes = self.all_default_boxes(device)  # [N, 4] cxcywh
        scores = F.softmax(cls_pred, dim=-1)             # [B, N, C]

        # default_boxes → xyxy 一次（供所有图复用）
        d_xyxy = _cxcywh_to_xyxy(default_boxes)

        results: list[list[dict]] = []
        for b in range(B):
            # 跳过背景类（index 0），取 class 1..C-1
            cls_scores = scores[b, :, 1:]                # [N, C-1]
            # 过滤：每 anchor 每类的最高分是否过阈值 —— 这里对全 (anchor,class) 过滤
            mask = cls_scores > score_threshold
            if not mask.any():
                results.append([])
                continue

            a_idx, c_idx = mask.nonzero(as_tuple=True)   # (K,)
            kept_scores = cls_scores[a_idx, c_idx]
            kept_locs = loc_pred[b, a_idx, :]
            kept_boxes = d_xyxy[a_idx]
            cls_ids = c_idx + 1                           # 恢复含背景的类别 id（1..C-1）

            # pre-NMS topk 截断
            if kept_scores.numel() > pre_nms_topk:
                topk = torch.topk(kept_scores, pre_nms_topk)
                sel = topk.indices
                kept_scores = kept_scores[sel]
                kept_locs = kept_locs[sel]
                kept_boxes = kept_boxes[sel]
                cls_ids = cls_ids[sel]

            # 解码 + 批量按类 NMS
            decoded = self._decode_xy(kept_locs, kept_boxes, variance)
            keep = batched_nms(decoded, kept_scores, cls_ids, nms_threshold)

            # 后处理：转 xywh + 限制 max_detections
            if keep.numel() > max_detections:
                keep = keep[torch.topk(kept_scores[keep], max_detections).indices]

            dets: list[dict] = []
            for idx in keep:
                x1, y1, x2, y2 = decoded[idx].tolist()
                # 归一化 [0,1] → 像素坐标（input_size 空间）
                # COCO eval 期望绝对像素坐标，非归一化值
                s = self.input_size
                bx = x1 * s
                by = y1 * s
                bw = (x2 - x1) * s
                bh = (y2 - y1) * s
                # 裁剪到有效范围
                bx = max(0.0, bx)
                by = max(0.0, by)
                bw = min(bw, s - bx)
                bh = min(bh, s - by)
                if bw <= 1.0 or bh <= 1.0:
                    continue
                dets.append({
                    "bbox": [bx, by, bw, bh],
                    "score": float(kept_scores[idx].item()),
                    "category_id": int(cls_ids[idx].item()) - 1,  # 内部 id (0-based)
                })
            results.append(dets)
        return results

    def _decode_xy(self, loc: torch.Tensor, default_boxes_xyxy: torch.Tensor,
                   variance: tuple[float, float]) -> torch.Tensor:
        """从 xyxy 格式 default box 解码 —— 为兼容 batched_nms，default box 已转 xyxy。

        _decode_single 假设 default_boxes 为 cxcywh；此处在一次性 xyxy 下重写解码，
        避免重复 cxcywh<->xyxy 转换。
        """
        x1, y1, x2, y2 = default_boxes_xyxy.unbind(-1)
        dcx = (x1 + x2) / 2
        dcy = (y1 + y2) / 2
        dw = x2 - x1
        dh = y2 - y1
        cx = dcx + loc[:, 0] * variance[0] * dw
        cy = dcy + loc[:, 1] * variance[0] * dh
        w = dw * torch.exp(loc[:, 2] * variance[1])
        h = dh * torch.exp(loc[:, 3] * variance[1])
        return torch.stack([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2,
        ], dim=-1)

    def _decode_single(self, loc: torch.Tensor, default_boxes: torch.Tensor,
                       variance: tuple[float, float]) -> torch.Tensor:
        cx = default_boxes[:, 0] + loc[:, 0] * variance[0] * default_boxes[:, 2]
        cy = default_boxes[:, 1] + loc[:, 1] * variance[0] * default_boxes[:, 3]
        w = default_boxes[:, 2] * torch.exp(loc[:, 2] * variance[1])
        h = default_boxes[:, 3] * torch.exp(loc[:, 3] * variance[1])
        return torch.stack([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2,
        ], dim=-1)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    from torchvision.ops import nms
    return nms(boxes, scores, iou_threshold)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """cx,cy,w,h → x1,y1,x2,y2。"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
