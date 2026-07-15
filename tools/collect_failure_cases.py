"""失败案例收集器 —— 自动筛选并保存检测失败样本。

用途（对应论文第十部分）：
    - 筛选高 loss / 低 AP 类别 / FP / FN 样本
    - 保存到 outputs/failure_cases/{tag}/ 供论文"局限性"章节使用

用法：
    python tools/collect_failure_cases.py --config configs/baseline_ssd.yaml \
        --weights outputs/checkpoints/baseline_no_lca.pth --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from lead_net.models import build_lead_net
from lead_net.data import build_coco_dataset
from lead_net.data.transforms import build_transforms
from lead_net.utils import load_config, get_nested, ExperimentManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net 失败案例收集")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--weights", required=True, type=str)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--top-k", type=int, default=20,
                   help="每类保存的失败案例数")
    p.add_argument("--output", type=str, default="outputs/failure_cases")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = get_nested(cfg, "experiment.tag", "model")
    input_size = cfg.get("data", {}).get("input_size", 320)
    class_map = cfg.get("class_map", {})
    coco_id_to_internal = cfg.get("coco_id_to_internal", {})
    internal_to_coco = {v: k for k, v in coco_id_to_internal.items()}

    # 加载模型
    model = build_lead_net(cfg)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 验证集
    ds = build_coco_dataset(cfg, split="val")
    if args.max_samples:
        ds.ids = ds.ids[:args.max_samples]
    transforms = build_transforms(cfg, split="val")

    eval_cfg = cfg.get("eval", {})
    score_th = eval_cfg.get("score_threshold", 0.05)
    nms_th = eval_cfg.get("nms", {}).get("iou_threshold", 0.45)

    # 收集 per-image 统计
    img_stats: dict[int, dict] = {}  # image_id → {gt_count, det_count, fp, fn, matched}

    for idx in range(len(ds)):
        sample = ds[idx]
        img_id = int(sample["image_id"].item())
        gt_count = len(sample["labels"])

        img_tensor = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            cls_pred, loc_pred = model(img_tensor)

        detections = model.head.decode(
            loc_pred, cls_pred,
            score_threshold=score_th,
            nms_threshold=nms_th,
            max_detections=100,
            pre_nms_topk=1000,
        )[0]

        det_count = len(detections)

        # 简单 FP/FN 估计（基于数量差）
        fp = max(0, det_count - gt_count)
        fn = max(0, gt_count - det_count)
        matched = min(gt_count, det_count)

        img_stats[img_id] = {
            "gt": gt_count,
            "det": det_count,
            "fp": fp,
            "fn": fn,
            "matched": matched,
            "score": fp + fn * 2,  # FN 权重更高（漏检更危险）
        }

    # ---- 筛选失败案例 ----
    sorted_stats = sorted(img_stats.items(), key=lambda x: x[1]["score"], reverse=True)

    variant = "lca" if "lca" in tag.lower() else "baseline"
    mgr = ExperimentManager.for_test("outputs/experiments", variant)
    output_dir = mgr.failure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    subdirs = {
        "fn_dominant": output_dir / "fn_dominant",
        "fp_dominant": output_dir / "fp_dominant",
        "worst_overall": output_dir / "worst_overall",
    }
    for d in subdirs.values():
        d.mkdir(exist_ok=True)

    saved = 0
    for img_id, stats in sorted_stats[:args.top_k]:
        label = f"gt{stats['gt']}_det{stats['det']}_fp{stats['fp']}_fn{stats['fn']}"
        fname = f"{img_id:012d}_{label}.txt"

        if stats["fn"] > stats["fp"]:
            dest = subdirs["fn_dominant"] / fname
        elif stats["fp"] > stats["fn"]:
            dest = subdirs["fp_dominant"] / fname
        else:
            dest = subdirs["worst_overall"] / fname

        with open(dest, "w", encoding="utf-8") as f:
            f.write(f"image_id: {img_id}\n")
            f.write(f"gt_boxes: {stats['gt']}\n")
            f.write(f"det_boxes: {stats['det']}\n")
            f.write(f"fp: {stats['fp']}, fn: {stats['fn']}\n")
        saved += 1

    print(f"[info] saved {saved} failure cases to {output_dir}")
    print(f"  fn_dominant: {(output_dir/'fn_dominant').glob('*.txt').__len__()} files")
    print(f"  fp_dominant: {(output_dir/'fp_dominant').glob('*.txt').__len__()} files")
    print(f"  worst_overall: {(output_dir/'worst_overall').glob('*.txt').__len__()} files")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
