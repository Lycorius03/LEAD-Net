"""TXT 数据集统计 + 锚框匹配诊断。

对 lead_subset 数据集的每个类别统计：
    - 目标数量分布
    - 尺寸分布 (small/medium/large)
    - 每图锚框正样本覆盖率

用法:
    python tools/dataset_stats_txt.py --config configs/train_lca.yaml
    python tools/dataset_stats_txt.py --config configs/train_lca.yaml --max-samples 500
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

# 修复 Windows GBK 终端 Unicode 输出问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lead_net.utils import load_config
from lead_net.data import build_dataloader
from lead_net.data.dataloader import collate_fn
from lead_net.models.detection_head import build_detection_head
from lead_net.models.loss import MultiBoxLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TXT 数据集统计与锚框匹配诊断")
    p.add_argument("--config", required=True, help="配置文件路径")
    p.add_argument("--max-samples", type=int, default=None,
                   help="限制分析样本数")
    p.add_argument("--split", type=str, default="train",
                   help="数据集分割 (train/val)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    class_map: dict = cfg.get("class_map", {})
    input_size = cfg.get("data", {}).get("input_size", 320)
    num_classes = cfg.get("num_classes", 7)
    device = torch.device("cpu")

    # ── 构建检测头 → default boxes ──
    head = build_detection_head(cfg)
    default_boxes = head.all_default_boxes(device)  # [N, 4] cxcywh normalized

    # ── 构建 dataloader ──
    loader = build_dataloader(cfg, split=args.split, batch_size=1,
                              num_workers=0, shuffle=False)
    ds = loader.dataset

    if args.max_samples:
        import random
        n = min(args.max_samples, len(ds))
        indices = random.Random(42).sample(range(len(ds)), n)
        ds = Subset(ds, indices)
        print(f"[info] 采样 {n}/{len(loader.dataset)} 张图片")

    print(f"=== TXT 数据集统计: {args.split} ({len(ds)} images) ===\n")

    # ── 统计容器 ──
    per_class_counts: dict[int, int] = defaultdict(int)
    per_class_img_counts: dict[int, set] = defaultdict(set)  # 类别出现的图片数
    all_widths: list[float] = []
    all_heights: list[float] = []
    all_areas: list[float] = []
    pos_counts: list[int] = []
    empty_img_count = 0
    total_gts = 0
    per_class_matched: dict[int, int] = defaultdict(int)  # 有正样本匹配的 GT 数
    per_class_total: dict[int, int] = defaultdict(int)    # 各类 GT 总数

    # ── 遍历数据集 ──
    from torchvision.ops import box_iou
    d_xyxy = _cxcywh_to_xyxy(default_boxes)

    for img_idx, batch in enumerate(loader):
        boxes_gt = batch["boxes"][0]   # [K, 4] XYWH pixel (post-resize)
        labels_gt = batch["labels"][0]  # [K]

        if boxes_gt.numel() == 0:
            empty_img_count += 1
            pos_counts.append(0)
            continue

        n_gt = boxes_gt.shape[0]
        total_gts += n_gt

        # 统计 GT 尺寸
        ws = boxes_gt[:, 2].tolist()
        hs = boxes_gt[:, 3].tolist()
        all_widths.extend(ws)
        all_heights.extend(hs)
        all_areas.extend([w * h for w, h in zip(ws, hs)])

        # 统计各类别目标数量
        for lbl in labels_gt.tolist():
            cls_id = int(lbl)
            per_class_counts[cls_id] += 1
            per_class_img_counts[cls_id].add(img_idx)
            per_class_total[cls_id] = per_class_total.get(cls_id, 0) + 1

        # 锚框匹配分析
        gtb_norm = boxes_gt.float() / input_size
        gtb_xyxy = _xywh_to_xyxy(gtb_norm)
        ious = box_iou(d_xyxy, gtb_xyxy)
        best_per_gt, best_anchor_per_gt = ious.max(dim=0)

        # 统计正样本数
        pos = best_per_gt >= 0.5
        pos_counts.append(pos.sum().item())

        # 各类别匹配统计
        for gt_idx in range(n_gt):
            cls_id = int(labels_gt[gt_idx].item())
            if best_per_gt[gt_idx] >= 0.5:
                per_class_matched[cls_id] = per_class_matched.get(cls_id, 0) + 1

        if (img_idx + 1) % 200 == 0:
            print(f"  processed {img_idx + 1}/{len(ds)} images...", flush=True)

    # ── 打印报告 ──
    print(f"\n{'='*60}")
    print(f"  类别分布")
    print(f"{'='*60}")
    print(f"{'ID':<5} {'Name':<15} {'GT count':>10} {'Images':>8} {'% matched':>10}")
    print("-" * 50)
    for cls_id in sorted(per_class_total.keys()):
        name = class_map.get(str(cls_id), f"cls_{cls_id}")
        count = per_class_counts.get(cls_id, 0)
        n_imgs = len(per_class_img_counts.get(cls_id, set()))
        n_matched = per_class_matched.get(cls_id, 0)
        n_total = per_class_total.get(cls_id, 1)
        match_pct = n_matched / max(n_total, 1) * 100
        print(f"{cls_id:<5} {name:<15} {count:>10} {n_imgs:>8} {match_pct:>9.1f}%")

    empty_pct = empty_img_count / max(len(ds), 1) * 100
    print(f"\n  空标注图片: {empty_img_count} ({empty_pct:.1f}%)")

    # ── 尺寸分布 ──
    if all_areas:
        areas_arr = np.array(all_areas)
        ws_arr = np.array(all_widths)
        hs_arr = np.array(all_heights)
        ratios = ws_arr / (hs_arr + 1e-8)

        small = (areas_arr < 32**2).sum()
        medium = ((areas_arr >= 32**2) & (areas_arr < 96**2)).sum()
        large = (areas_arr >= 96**2).sum()
        n_total = len(areas_arr) or 1
        print(f"\n{'='*60}")
        print(f"  目标尺寸分布 (n={n_total})")
        print(f"{'='*60}")
        print(f"  Small  (<1024 px²):  {small:>6} ({small/n_total*100:.1f}%)")
        print(f"  Medium (1024-9216):  {medium:>6} ({medium/n_total*100:.1f}%)")
        print(f"  Large  (>9216 px²):  {large:>6} ({large/n_total*100:.1f}%)")
        print(f"\n  宽度:  P5={np.percentile(ws_arr,5):.0f}  P25={np.percentile(ws_arr,25):.0f}  "
              f"P50={np.percentile(ws_arr,50):.0f}  P75={np.percentile(ws_arr,75):.0f}  "
              f"P95={np.percentile(ws_arr,95):.0f}")
        print(f"  高度:  P5={np.percentile(hs_arr,5):.0f}  P25={np.percentile(hs_arr,25):.0f}  "
              f"P50={np.percentile(hs_arr,50):.0f}  P75={np.percentile(hs_arr,75):.0f}  "
              f"P95={np.percentile(hs_arr,95):.0f}")
        print(f"  宽高比: P5={np.percentile(ratios,5):.2f}  P25={np.percentile(ratios,25):.2f}  "
              f"P50={np.percentile(ratios,50):.2f}  P75={np.percentile(ratios,75):.2f}  "
              f"P95={np.percentile(ratios,95):.2f}")

    # ── 锚框匹配 ──
    if pos_counts:
        pos_arr = np.array(pos_counts)
        print(f"\n{'='*60}")
        print(f"  锚框正样本匹配统计")
        print(f"{'='*60}")
        print(f"  每图正样本数: 均值={pos_arr.mean():.1f}  "
              f"中位数={np.median(pos_arr):.0f}  "
              f"P25={np.percentile(pos_arr,25):.0f}  "
              f"P75={np.percentile(pos_arr,75):.0f}")
        pct_zero = (pos_arr == 0).mean() * 100
        print(f"  零正样本图片: {pct_zero:.1f}%")
        print(f"  无GT图片:     {empty_pct:.1f}%")
        if pct_zero > empty_pct + 5:
            print(f"  ⚠️  有 GT 但匹配失败: ~{pct_zero - empty_pct:.1f}% — 锚框可能不适合此数据集")

        # 正样本不足分析
        very_few = (pos_arr > 0) & (pos_arr < 5)
        pct_few = very_few.mean() * 100
        if pct_few > 10:
            print(f"  ⚠️  正样本<5 的图片: {pct_few:.1f}% — 建议降低 IoU threshold 或调整 anchor")

    # ── 锚框覆盖范围 ──
    d_w = default_boxes[:, 2] * input_size
    d_h = default_boxes[:, 3] * input_size
    print(f"\n  锚框覆盖范围 ({len(default_boxes)} 个):")
    print(f"    宽度:  [{d_w.min().item():.0f}, {d_w.max().item():.0f}] px")
    print(f"    高度:  [{d_h.min().item():.0f}, {d_h.max().item():.0f}] px")
    d_areas = (d_w * d_h)
    d_ratios = d_w / (d_h + 1e-8)
    print(f"    面积:  [{d_areas.min().item():.0f}, {d_areas.max().item():.0f}] px²")
    print(f"    宽高比: [{d_ratios.min().item():.2f}, {d_ratios.max().item():.2f}]")

    print(f"\n{'='*60}")
    print(f"  诊断完成。")
    print(f"{'='*60}")

    return 0


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


if __name__ == "__main__":
    raise SystemExit(main())
