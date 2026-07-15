"""数据集统计工具 —— 类别分布 / 目标尺度 / 宽高比。

用途（对应论文"数据集统计信息"）：
    - train/val 每个类别的样本数量分布
    - 目标尺度分布 (small<32² / medium 32²-96² / large>96²)
    - bbox 宽高比分布

用法：
    python tools/dataset_stats.py --config configs/baseline_ssd.yaml
    python tools/dataset_stats.py --config configs/baseline_ssd.yaml --split train
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lead_net.utils import load_config
from lead_net.data import build_coco_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net 数据集统计分析")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--split", type=str, default="train",
                   help="train 或 val")
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    class_map: dict = cfg.get("class_map", {})

    ds = build_coco_dataset(cfg, split=args.split)
    if args.max_samples:
        ds.ids = ds.ids[:args.max_samples]

    print(f"=== Dataset Statistics: {args.split} ({len(ds)} images) ===")
    print()

    # ---- 收集统计 ----
    per_class_counts: dict[int, int] = defaultdict(int)
    all_widths: list[float] = []
    all_heights: list[float] = []
    all_areas: list[float] = []

    for idx in range(len(ds)):
        sample = ds[idx]
        labels = sample["labels"].tolist()
        boxes = sample["boxes"]  # BoundingBoxes [K, 4] XYWH

        for lbl in labels:
            per_class_counts[int(lbl)] += 1

        if boxes.shape[0] > 0:
            ws = boxes[:, 2].tolist()
            hs = boxes[:, 3].tolist()
            all_widths.extend(ws)
            all_heights.extend(hs)
            all_areas.extend([w * h for w, h in zip(ws, hs)])

    # ---- 类别分布 ----
    print("--- Per-Class Sample Count ---")
    print(f"{'class_id':<10} {'class_name':<20} {'count':>8}")
    print("-" * 40)
    for cls_id in sorted(per_class_counts.keys()):
        name = class_map.get(str(cls_id), f"cls_{cls_id}")
        print(f"{cls_id:<10} {name:<20} {per_class_counts[cls_id]:>8}")

    # ---- 尺度分布 ----
    ws_arr = np.array(all_widths)
    hs_arr = np.array(all_heights)
    areas_arr = np.array(all_areas)
    ratios = ws_arr / (hs_arr + 1e-8)

    print(f"\n--- Object Scale Distribution (n={len(all_areas)}) ---")
    small = (areas_arr < 32**2).sum()
    medium = ((areas_arr >= 32**2) & (areas_arr < 96**2)).sum()
    large = (areas_arr >= 96**2).sum()
    n_total = len(all_areas) or 1
    print(f"  Small  (<1024 px^2):  {small:>6} ({small/n_total*100:.1f}%)")
    print(f"  Medium (1024-9216):   {medium:>6} ({medium/n_total*100:.1f}%)")
    print(f"  Large  (>9216 px^2):  {large:>6} ({large/n_total*100:.1f}%)")

    print(f"\n--- BBox Dimension Percentiles (pixels) ---")
    for p in [5, 25, 50, 75, 95]:
        print(f"  Width  P{p:>2}: {np.percentile(ws_arr, p):.1f}")
    for p in [5, 25, 50, 75, 95]:
        print(f"  Height P{p:>2}: {np.percentile(hs_arr, p):.1f}")

    print(f"\n--- Aspect Ratio Distribution ---")
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p:>2}: {np.percentile(ratios, p):.3f}")

    # ---- CSV 输出 ----
    csv_path = Path("outputs/experiments") / f"dataset_statistics_{args.split}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("split,class_id,class_name,sample_count\n")
        for cls_id, count in sorted(per_class_counts.items()):
            name = class_map.get(str(cls_id), f"cls_{cls_id}")
            f.write(f"{args.split},{cls_id},{name},{count}\n")
    print(f"\n[info] CSV saved to {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
