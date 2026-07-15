"""评估入口脚本（独立于训练）。

用法：
    python tools/eval.py --config configs/baseline_ssd.yaml --weights outputs/checkpoints/last.pth
    python tools/eval.py --config configs/lca_ssd.yaml --weights outputs/checkpoints/baseline_plus_lca.pth --max-samples 50 --score-threshold 0.05

依据：
    - docs/ARCHITECTURE.md §配置驱动；docs/EXPERIMENTS.md：mAP 指标用于消融实验。
    - 模块化原则：训练归 tools/train.py，评估归本脚本，推理归 tools/infer.py。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lead_net.data import build_dataloader
from lead_net.engine import Evaluator
from lead_net.models import build_lead_net
from lead_net.utils import load_config, resolve_paths_in, get_nested, ExperimentManager, ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net 评估入口：加载 checkpoint 在验证集上算 COCO mAP")
    p.add_argument("--config", required=True, type=str, help="配置文件路径（需与训练时一致）")
    p.add_argument("--weights", required=True, type=str,
                   help="checkpoint 路径（如 outputs/checkpoints/last.pth）")
    p.add_argument("--max-samples", type=int, default=None,
                   help="限制验证集样本数（用于快速测试）")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"],
                   help="评估的数据集划分（默认 val）")
    p.add_argument("--score-threshold", type=float, default=None,
                   help="覆盖配置中的 eval.score_threshold（推理置信度阈值）")
    p.add_argument("--nms-iou", type=float, default=None,
                   help="覆盖配置中的 eval.nms.iou_threshold")
    p.add_argument("--max-detections", type=int, default=None,
                   help="覆盖每图保留的最大检测数")
    return p.parse_args()


def _limit_dataset(ds, n: int):
    total = len(ds)
    n = min(n, total)
    ds.ids = [ds.ids[i] for i in range(n)]
    print(f"[eval] dataset limited from {total} to {n} samples")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = resolve_paths_in(cfg)

    # 命令行覆盖评估阈值
    eval_cfg = cfg.setdefault("eval", {})
    if args.score_threshold is not None:
        eval_cfg["score_threshold"] = args.score_threshold
    if args.nms_iou is not None:
        eval_cfg.setdefault("nms", {})["iou_threshold"] = args.nms_iou
    if args.max_detections is not None:
        eval_cfg["nms"] = {**(eval_cfg.get("nms", {})), "max_detections": args.max_detections}

    dev_type = get_nested(cfg, "device.type", "cuda")
    device = torch.device(dev_type if (dev_type == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[eval] device: {device}")

    # 数据
    num_workers = 0 if sys.platform == "win32" else get_nested(cfg, "train.num_workers", 4)
    val_loader = build_dataloader(cfg, split=args.split, num_workers=num_workers, shuffle=False)
    if args.max_samples:
        _limit_dataset(val_loader.dataset, args.max_samples)
    print(f"[eval] {args.split} samples: {len(val_loader.dataset)} | "
          f"score_thr={eval_cfg.get('score_threshold', 0.01)} "
          f"nms_iou={eval_cfg.get('nms', {}).get('iou_threshold', 0.45)} "
          f"max_det={eval_cfg.get('nms', {}).get('max_detections', 100)}")

    # 模型 + 权重
    model = build_lead_net(cfg).to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"[eval] loaded weights: {args.weights} (tag={ckpt.get('tag') if isinstance(ckpt, dict) else 'N/A'})")

    # 实验管理：找到最新 session 和对应 variant
    tag = ckpt.get("tag") if isinstance(ckpt, dict) else "eval"
    variant = "lca" if "lca" in tag.lower() else "baseline"
    session = ExperimentManager.latest_session("outputs/experiments") or "exp_001"
    mgr = ExperimentManager("outputs/experiments", session, variant)
    mgr.eval_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = mgr.eval_dir

    # 评估
    evaluator = Evaluator(model=model, val_loader=val_loader, cfg=cfg, device=device)
    metrics = evaluator.evaluate()
    print(f"[eval] mAP@0.5={metrics.get('mAP@0.5', 0.0):.4f} "
          f"mAP@0.5:0.95={metrics.get('mAP@0.5:0.95', 0.0):.4f} "
          f"mAP@0.75={metrics.get('mAP@0.75', 0.0):.4f}")

    # 深度分析（PR曲线、混淆矩阵、per-IoU AP）→ eval/ 子目录
    predictions = _collect_predictions(model, val_loader, cfg, device)
    if predictions:
        from lead_net.engine.eval_analysis import analyze_coco_eval
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import json, tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(predictions, f)
            pred_path = f.name
        try:
            coco_gt = val_loader.dataset.coco
            coco_dt = coco_gt.loadRes(pred_path)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            analyze_coco_eval(coco_eval, cfg, eval_dir, tag)
        finally:
            Path(pred_path).unlink(missing_ok=True)
    return 0


def _collect_predictions(model, val_loader, cfg, device):
    """收集所有预测结果（复用 Evaluator 逻辑但不打印）。"""
    predictions = []
    coco_id_to_internal = cfg.get("coco_id_to_internal", {})
    internal_to_coco = {v: k for k, v in coco_id_to_internal.items()}
    eval_cfg = cfg.get("eval", {})
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            cls_pred, loc_pred = model(images)
            detections = model.head.decode(
                loc_pred, cls_pred,
                score_threshold=eval_cfg.get("score_threshold", 0.05),
                nms_threshold=eval_cfg.get("nms", {}).get("iou_threshold", 0.45),
                max_detections=eval_cfg.get("nms", {}).get("max_detections", 100),
                pre_nms_topk=eval_cfg.get("nms", {}).get("pre_nms_topk", 1000),
            )
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
    return predictions


if __name__ == "__main__":
    raise SystemExit(main())