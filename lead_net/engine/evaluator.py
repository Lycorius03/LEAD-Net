"""评估器 —— mAP / per-class AP / precision / recall。

依据：
    - docs/EXPERIMENTS.md：mAP@0.5 用于消融实验。
    - 工业标准（YOLO/COCO）：mAP@0.5:0.95, per-class AP, precision, recall。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Evaluator:
    """目标检测评估器。

    返回 COCO 标准指标：
        - mAP@0.5 / mAP@0.5:0.95 / mAP@0.75
        - per-class AP@0.5 / AP@0.5:0.95 / AP@0.75
        - precision / recall (from COCO eval stats)
    """

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device | None = None,
    ):
        self.model = model
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """计算完整 COCO 指标。"""
        self.model.eval()
        eval_cfg = self.cfg.get("eval", {})
        score_threshold = eval_cfg.get("score_threshold", 0.05)
        nms_threshold = eval_cfg.get("nms", {}).get("iou_threshold", 0.45)
        max_detections = eval_cfg.get("nms", {}).get("max_detections", 100)
        pre_nms_topk = eval_cfg.get("nms", {}).get("pre_nms_topk", 1000)

        # internal id → COCO id 映射
        coco_id_to_internal = self.cfg.get("coco_id_to_internal", {})
        internal_to_coco = {v: k for k, v in coco_id_to_internal.items()}

        predictions = []
        n_batches = len(self.val_loader)
        for bi, batch in enumerate(self.val_loader):
            images = batch["image"].to(self.device)

            cls_pred, loc_pred = self.model(images)

            detections = self.model.head.decode(
                loc_pred, cls_pred,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
                max_detections=max_detections,
                pre_nms_topk=pre_nms_topk,
            )
            if (bi + 1) % 10 == 0 or bi + 1 == n_batches:
                print(f"[eval] batch {bi+1}/{n_batches}", flush=True)

            for i, dets in enumerate(detections):
                img_id = int(batch["image_id"][i].item())
                for det in dets:
                    coco_cat = internal_to_coco.get(det["category_id"])
                    if coco_cat is None:
                        continue
                    predictions.append({
                        "image_id": img_id,
                        "category_id": coco_cat,
                        "bbox": [float(x) for x in det["bbox"]],
                        "score": float(det["score"]),
                    })

        if not predictions:
            print("[eval] 无预测结果，可能所有置信度低于阈值")
            return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0, "mAP@0.75": 0.0}

        return _coco_eval_full(predictions, self.val_loader.dataset, self.cfg)


def _coco_eval_full(
    predictions: list[dict], dataset, cfg: dict,
) -> dict[str, Any]:
    """使用 pycocotools 计算完整 COCO 指标（含 per-class AP）。"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(predictions, f)
        pred_path = f.name

    try:
        coco_gt = dataset.coco
        coco_dt = coco_gt.loadRes(pred_path)

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        result: dict[str, Any] = {
            "mAP@0.5:0.95": float(stats[0]),
            "mAP@0.5": float(stats[1]),
            "mAP@0.75": float(stats[2]),
            # COCO 还提供按尺度分组的 AP
            "AP_small": float(stats[3]) if len(stats) > 3 else None,
            "AP_medium": float(stats[4]) if len(stats) > 4 else None,
            "AP_large": float(stats[5]) if len(stats) > 5 else None,
            # AR (Average Recall)
            "AR_max1": float(stats[6]) if len(stats) > 6 else None,
            "AR_max10": float(stats[7]) if len(stats) > 7 else None,
            "AR_max100": float(stats[8]) if len(stats) > 8 else None,
        }

        # per-class AP
        result["per_class"] = _extract_per_class_ap(coco_eval, cfg)
        return result
    finally:
        Path(pred_path).unlink(missing_ok=True)


def _extract_per_class_ap(coco_eval, cfg: dict) -> list[dict[str, Any]]:
    """从 COCOeval 对象提取 per-class AP@0.5 / AP@0.5:0.95 / AP@0.75。

    pycocotools 内部结构：
        coco_eval.eval["precision"]: [T, R, K, A, M]
            T = 10 IoU thresholds (0.5:0.05:0.95)
            R = 101 recall thresholds
            K = num categories
            A = 4 area ranges (all/small/medium/large)
            M = 3 max detections (1/10/100)
    """
    class_map: dict = cfg.get("class_map", {})
    coco_id_to_internal: dict = cfg.get("coco_id_to_internal", {})

    precision = coco_eval.eval.get("precision")
    if precision is None or precision.size == 0:
        return []

    # COCOeval.params.catIds 记录了 precision 第 2 维对应的 COCO category ID 顺序
    cat_ids = coco_eval.params.catIds
    results = []
    num_classes = precision.shape[2]

    for cls_idx in range(num_classes):
        if cls_idx >= len(cat_ids):
            continue
        coco_cat_id = cat_ids[cls_idx]
        internal_id = coco_id_to_internal.get(str(coco_cat_id), coco_id_to_internal.get(coco_cat_id))
        if internal_id is None:
            continue
        class_name = class_map.get(str(internal_id), f"cls_{internal_id}")

        # AP@0.5: IoU idx=0 (threshold=0.5), area=0 (all), max_dets=2 (100)
        ap50 = float(precision[0, :, cls_idx, 0, 2].mean())
        # AP@0.5:0.95: IoU idx=0..9 mean, area=0, max_dets=2
        ap_all = float(precision[:, :, cls_idx, 0, 2].mean())
        # AP@0.75: IoU idx=5 (threshold=0.75), area=0, max_dets=2
        ap75 = float(precision[5, :, cls_idx, 0, 2].mean()) if precision.shape[0] > 5 else None

        results.append({
            "class_id": internal_id,
            "class_name": class_name,
            "AP@0.5": ap50,
            "AP@0.5:0.95": ap_all,
            "AP@0.75": ap75,
        })

    return results
