"""锚框分析工具 —— 统计 COCO GT box 分布与 default box IoU 匹配。

用途（M3-B）：
    - 统计 val2017 中所有 GT box 的宽高分布（像素坐标，input_size=320 归一化后）。
    - 计算每个 GT box 与 2075 个 default box 的最佳 IoU。
    - 输出覆盖率报告，仅在结构性失配时建议调整 anchor 配置。

用法：
    python tools/inspect_anchors.py --config configs/baseline_ssd.yaml
    python tools/inspect_anchors.py --config configs/baseline_ssd.yaml --max-samples 100
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from lead_net.data import build_coco_dataset
from lead_net.models.detection_head import build_detection_head
from lead_net.utils import load_config, get_nested


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net 锚框分析（M3-B）")
    p.add_argument("--config", required=True, type=str,
                   help="配置文件路径（如 configs/baseline_ssd.yaml）")
    p.add_argument("--max-samples", type=int, default=None,
                   help="限制分析样本数（默认全量 val2017）")
    p.add_argument("--iou-thresholds", type=float, nargs="+",
                   default=[0.5, 0.7],
                   help="覆盖率阈值列表（默认 0.5 0.7）")
    return p.parse_args()


def build_default_boxes_tensor(cfg: dict, device: torch.device) -> torch.Tensor:
    """从配置构建全部 default boxes，返回 [N, 4] (cxcywh, 像素坐标)。"""
    input_size = cfg.get("data", {}).get("input_size", 320)
    head = build_detection_head(cfg)  # 使用与模型一致的工厂函数
    boxes = head.all_default_boxes(device)  # normalized cxcywh
    boxes_px = boxes.clone()
    boxes_px[:, :4] *= input_size
    return boxes_px


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """cx,cy,w,h → x1,y1,x2,y2 (pixel coords)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def compute_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """向量化 IoU：boxes_a [M,4] xyxy, boxes_b [N,4] xyxy → [M, N]."""
    # 扩展维度广播
    a = boxes_a.unsqueeze(1)  # [M, 1, 4]
    b = boxes_b.unsqueeze(0)  # [1, N, 4]

    inter_x1 = torch.max(a[..., 0], b[..., 0])
    inter_y1 = torch.max(a[..., 1], b[..., 1])
    inter_x2 = torch.min(a[..., 2], b[..., 2])
    inter_y2 = torch.min(a[..., 3], b[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-8)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    input_size = cfg.get("data", {}).get("input_size", 320)
    device = torch.device("cpu")  # 锚框分析用 CPU 足够

    # ---- 构建 default boxes ----
    dboxes = build_default_boxes_tensor(cfg, device)  # [2075, 4] cxcywh px
    dboxes_xyxy = cxcywh_to_xyxy(dboxes)
    print(f"[info] total default boxes: {dboxes.shape[0]}")
    print(f"[info] input_size: {input_size}")

    # ---- 构建数据集（val split，不应用训练增强） ----
    ds = build_coco_dataset(cfg, split="val")
    if args.max_samples:
        total = len(ds)
        n = min(args.max_samples, total)
        ds.ids = ds.ids[:n]
        print(f"[info] dataset limited from {total} to {n} samples")

    print(f"[info] val samples: {len(ds)}")

    # ---- 统计容器 ----
    gt_widths: list[float] = []
    gt_heights: list[float] = []
    gt_areas: list[float] = []
    best_ious: list[float] = []
    per_class_best_iou: dict[int, list[float]] = defaultdict(list)
    per_scale_best_iou: dict[str, list[float]] = defaultdict(list)
    unmatched: list[dict] = []  # 记录 best_iou < 0.3 的 GT

    # ---- 遍历数据集 ----
    num_workers = 0 if sys.platform == "win32" else 2
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers,
                        collate_fn=lambda x: x)

    total_gts = 0
    for batch in loader:
        for sample in batch:
            boxes_gt = sample["boxes"]   # BoundingBoxes [K, 4] XYWH px (pre-resize)
            labels = sample["labels"]     # [K]

            if boxes_gt.shape[0] == 0:
                continue

            # GT 转为 xyxy 像素坐标
            gt_xywh = boxes_gt.clone().to(torch.float32)
            gt_xyxy = cxcywh_to_xyxy(gt_xywh)

            # 统计 GT 尺寸
            ws = gt_xywh[:, 2].tolist()
            hs = gt_xywh[:, 3].tolist()
            gt_widths.extend(ws)
            gt_heights.extend(hs)
            gt_areas.extend([w * h for w, h in zip(ws, hs)])

            # 计算每个 GT 与所有 default box 的 IoU
            ious = compute_iou(gt_xyxy, dboxes_xyxy)  # [K, 2075]
            batch_best, _ = ious.max(dim=1)
            batch_best_list = batch_best.tolist()
            best_ious.extend(batch_best_list)

            # 按类别分组
            for i, label in enumerate(labels.tolist()):
                per_class_best_iou[label].append(batch_best_list[i])

            # 按 GT 尺寸分组
            for i in range(len(ws)):
                area = ws[i] * hs[i]
                if area < 32 * 32:
                    scale = "small (<32px)"
                elif area < 96 * 96:
                    scale = "medium (32-96px)"
                elif area < 160 * 160:
                    scale = "large (96-160px)"
                else:
                    scale = "xlarge (>160px)"
                per_scale_best_iou[scale].append(batch_best_list[i])

            # 记录严重不匹配的
            for i in range(len(ws)):
                if batch_best_list[i] < 0.3:
                    unmatched.append({
                        "w": int(ws[i]), "h": int(hs[i]),
                        "area": int(ws[i] * hs[i]),
                        "best_iou": round(batch_best_list[i], 3),
                        "class": int(labels[i].item()),
                    })

            total_gts += len(ws)

    print(f"\n{'='*60}")
    print(f"  GT Box 分布统计（{total_gts} 个标注，{len(ds)} 张图片）")
    print(f"{'='*60}")

    # ---- GT 尺寸分布 ----
    ws_arr = np.array(gt_widths)
    hs_arr = np.array(gt_heights)
    areas_arr = np.array(gt_areas)
    ratios = ws_arr / (hs_arr + 1e-8)

    print(f"\n--- 宽度 (pixels) ---")
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p}: {np.percentile(ws_arr, p):.1f}")

    print(f"\n--- 高度 (pixels) ---")
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p}: {np.percentile(hs_arr, p):.1f}")

    print(f"\n--- 宽高比 (w/h) ---")
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p}: {np.percentile(ratios, p):.2f}")

    print(f"\n--- 面积 (px^2) ---")
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p}: {np.percentile(areas_arr, p):.0f}")

    # ---- Default box 覆盖统计 ----
    d_w = dboxes[:, 2]
    d_h = dboxes[:, 3]
    d_areas = (d_w * d_h).tolist()
    d_ratios = (d_w / (d_h + 1e-8)).tolist()
    d_min_w, d_max_w = d_w.min().item(), d_w.max().item()
    d_min_h, d_max_h = d_h.min().item(), d_h.max().item()

    print(f"\n--- Default Box 覆盖范围 (2075 个) ---")
    print(f"  宽度: [{d_min_w:.1f}, {d_max_w:.1f}] px")
    print(f"  高度: [{d_min_h:.1f}, {d_max_h:.1f}] px")
    print(f"  面积: [{min(d_areas):.0f}, {max(d_areas):.0f}] px^2")
    print(f"  宽高比: [{min(d_ratios):.2f}, {max(d_ratios):.2f}]")

    # ---- IoU 匹配统计 ----
    best_arr = np.array(best_ious)
    print(f"\n--- GT vs Default Box 最佳 IoU 分布 ---")
    print(f"  Mean:  {best_arr.mean():.4f}")
    print(f"  Median: {np.median(best_arr):.4f}")
    print(f"  Std:   {best_arr.std():.4f}")
    print(f"  Min:   {best_arr.min():.4f}")
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p}:   {np.percentile(best_arr, p):.4f}")

    print(f"\n--- 覆盖率 (GT best IoU > threshold) ---")
    for th in args.iou_thresholds:
        covered = (best_arr >= th).sum()
        rate = covered / len(best_arr) * 100
        print(f"  IoU > {th:.1f}: {covered}/{len(best_arr)} = {rate:.1f}%")

    # ---- 按尺度分组的覆盖率 ----
    print(f"\n--- 按 GT 尺度分组的覆盖率 ---")
    for scale in ["small (<32px)", "medium (32-96px)", "large (96-160px)", "xlarge (>160px)"]:
        ious_s = per_scale_best_iou.get(scale, [])
        if not ious_s:
            continue
        arr = np.array(ious_s)
        cov_05 = (arr >= 0.5).mean() * 100
        cov_07 = (arr >= 0.7).mean() * 100
        print(f"  {scale:22s}: n={len(arr):4d}  IoU>0.5={cov_05:.1f}%  IoU>0.7={cov_07:.1f}%  mean={arr.mean():.3f}")

    # ---- 严重不匹配项 ----
    print(f"\n--- 不匹配 (<0.3 IoU): {len(unmatched)} 个 ---")
    if unmatched:
        unmatched.sort(key=lambda x: x["best_iou"])
        for u in unmatched[:20]:
            cls_name = cfg.get("class_map", {}).get(str(u["class"]), f"cls_{u['class']}")
            print(f"  {u['w']:4d}x{u['h']:<5d} area={u['area']:6d}  best_iou={u['best_iou']:.3f}  class={cls_name}")
        if len(unmatched) > 20:
            print(f"  ... 还有 {len(unmatched) - 20} 个")

    # ---- 结论 ----
    print(f"\n{'='*60}")
    cov_05 = (best_arr >= 0.5).mean() * 100
    if cov_05 >= 70:
        print(f"  结论：锚框覆盖率 {cov_05:.1f}% >= 70%，配置合理，无需调整。")
    else:
        print(f"  结论：锚框覆盖率 {cov_05:.1f}% < 70%，建议核查锚框配置。")
        if unmatched:
            print(f"  提示：{len(unmatched)} 个 GT 的 best IoU < 0.3，")
            print(f"        主要集中在极小/极大目标或极端宽高比。")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
