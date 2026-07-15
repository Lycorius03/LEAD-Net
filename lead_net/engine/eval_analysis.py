"""评估深度分析工具 —— PR曲线 + 混淆矩阵 + per-IoU 分解。

从 pycocotools 评估结果中提取论文级详细指标。
独立于 Evaluator 核心逻辑，可单独调用。

输出文件（写入 outputs/experiments/）：
    - {tag}_pr_curve.csv        per-class PR 曲线数据
    - {tag}_confusion.csv       混淆矩阵
    - {tag}_per_iou_ap.csv      AP @ each IoU threshold
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def analyze_coco_eval(
    coco_eval,
    cfg: dict,
    output_dir: str | Path,
    tag: str,
) -> dict[str, Path]:
    """从 COCOeval 对象提取所有深度分析数据。

    Args:
        coco_eval: pycocotools.COCEval 实例（已调用 evaluate() + accumulate()）
        cfg: 配置
        output_dir: 输出目录
        tag: 实验标识

    Returns:
        {csv_type: file_path} dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. PR curve
    pr_path = output_dir / f"{tag}_pr_curve.csv"
    _export_pr_curves(coco_eval, cfg, pr_path)
    results["pr_curve"] = pr_path

    # 2. Confusion matrix
    cm_path = output_dir / f"{tag}_confusion.csv"
    _export_confusion(coco_eval, cfg, cm_path)
    results["confusion"] = cm_path

    # 3. Per-IoU AP
    iou_path = output_dir / f"{tag}_per_iou_ap.csv"
    _export_per_iou_ap(coco_eval, iou_path)
    results["per_iou_ap"] = iou_path

    return results


def _export_pr_curves(coco_eval, cfg: dict, path: Path) -> None:
    """导出 per-class PR 曲线数据。

    对每个类别，输出 (recall, precision) 点对。
    pycocotools 的 eval["precision"]: [T=10, R=101, K, A=4, M=3]
    取 area=0(all), max_dets=2(100)，在 IoU=0.50 (idx=0) 处展开。
    """
    class_map = cfg.get("class_map", {})
    cat_ids = coco_eval.params.catIds
    precision = coco_eval.eval.get("precision")

    if precision is None or precision.size == 0:
        return

    recalls = np.linspace(0, 1, precision.shape[1])  # 101 points

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("class_id,class_name,iou_type,recall,precision\n")
        for cls_idx, cat_id in enumerate(cat_ids):
            if cls_idx >= precision.shape[2]:
                continue
            name = class_map.get(str(cls_idx), f"cls_{cls_idx}")
            # IoU=0.5, area=all, max_dets=100 → idx (0, :, cls, 0, 2)
            pr = precision[0, :, cls_idx, 0, 2]
            for r, p in zip(recalls, pr):
                if p >= 0:
                    f.write(f"{cls_idx},{name},IoU0.50,{r:.4f},{p:.4f}\n")

    print(f"[analysis] PR curves → {path.name}")


def _export_confusion(coco_eval, cfg: dict, path: Path) -> None:
    """导出混淆矩阵。

    从 pycocotools 的 precision 矩阵中提取 per-IoU=0.5 时的 per-class AP，
    构造 (pred_class → true_class) 混淆近似：用各类别的 FN/FP 计数。

    注：pycocotools 不直接提供混淆矩阵，此处用每类 TP/FP/FN 推算。
    """
    class_map = cfg.get("class_map", {})
    cat_ids = coco_eval.params.catIds
    eval_data = coco_eval.eval

    if eval_data is None:
        return

    # 从 annotation 和 detection 计算 per-class TP/FP/FN
    # 使用 eval["counts"] 矩阵
    # counts[T,R,K,A,M] 对应 TP(0) / FN(1) / FP(2)
    counts = eval_data.get("counts")
    if counts is None or counts.size == 0:
        return

    n_classes = min(len(cat_ids), counts.shape[2])

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("class_id,class_name,TP,FN,FP,precision,recall\n")
        for cls_idx in range(n_classes):
            name = class_map.get(str(cls_idx), f"cls_{cls_idx}")
            # IoU=0.5, area=all, max_dets=100 → (0, :, cls, 0, 2)
            tp = counts[0, :, cls_idx, 0, 2].sum()
            fn = counts[1, :, cls_idx, 0, 2].sum()
            fp = counts[2, :, cls_idx, 0, 2].sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f.write(f"{cls_idx},{name},{tp:.0f},{fn:.0f},{fp:.0f},{prec:.4f},{rec:.4f}\n")

    print(f"[analysis] confusion matrix → {path.name}")


def _export_per_iou_ap(coco_eval, path: Path) -> None:
    """导出 per-IoU AP 分解。

    pycocotools 在 10 个 IoU 阈值 (0.50:0.05:0.95) 上评估，
    取每个阈值的平均 AP (area=all, max_dets=100)。
    """
    precision = coco_eval.eval.get("precision")
    if precision is None or precision.size == 0:
        return

    iou_thresholds = np.linspace(0.5, 0.95, precision.shape[0])

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("iou_threshold,AP\n")
        for t_idx, iou in enumerate(iou_thresholds):
            ap = precision[t_idx, :, :, 0, 2].mean()
            f.write(f"{iou:.2f},{ap:.6f}\n")

    print(f"[analysis] per-IoU AP → {path.name}")


def run_analysis_from_predictions(
    predictions: list[dict],
    dataset,
    cfg: dict,
    output_dir: str | Path = "outputs/experiments",
    tag: str = "model",
) -> dict[str, Path]:
    """从原始预测列表运行完整分析。

    这是 Evaluator.evaluate() 的补充——在已有 predictions 后调用此函数
    获取深度分析数据，而不仅仅是 mAP summary。

    Args:
        predictions: COCO 格式预测列表 [{"image_id":, "category_id":, "bbox":, "score":}, ...]
        dataset: COCODetection 数据集实例
        cfg: 配置
        output_dir: 输出目录
        tag: 实验标识

    Returns:
        {csv_type: file_path}
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                      encoding="utf-8") as f:
        json.dump(predictions, f)
        pred_path = f.name

    try:
        coco_gt = dataset.coco
        coco_dt = coco_gt.loadRes(pred_path)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        return analyze_coco_eval(coco_eval, cfg, output_dir, tag)
    finally:
        Path(pred_path).unlink(missing_ok=True)
