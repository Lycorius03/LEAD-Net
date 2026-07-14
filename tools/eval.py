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
from lead_net.utils import load_config, resolve_paths_in, get_nested, ensure_dir


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

    # 评估
    evaluator = Evaluator(model=model, val_loader=val_loader, cfg=cfg, device=device)
    metrics = evaluator.evaluate()
    print(f"[eval] mAP@0.5={metrics.get('mAP@0.5', 0.0):.4f} "
          f"mAP@0.5:0.95={metrics.get('mAP@0.5:0.95', 0.0):.4f} "
          f"mAP@0.75={metrics.get('mAP@0.75', 0.0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())