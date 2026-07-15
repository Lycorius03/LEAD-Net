"""INT8 量化精度对比工具 —— FP32 vs INT8 per-class AP 差异。

用途（对应 RQ3 / DATA_COLLECTION.md 第四类）：
    - 加载 FP32 权重和 INT8 TFLite 模型，分别在 val2017 上评估
    - 输出每个类别的 AP 变化（d_mAP），找出量化敏感类别
    - 记录校准集构成和量化耗时

使用前提：
    - M5 完成后：FP32 checkpoint + INT8 TFLite 模型均已产出
    - 需要 pycocotools + tensorflow（或 tflite_runtime）

用法：
    python tools/eval_quantization.py \
        --config configs/baseline_ssd.yaml \
        --fp32-weights outputs/checkpoints/baseline_no_lca.pth \
        --int8-model outputs/tflite/baseline_no_lca_int8.tflite \
        --calibration-dir data/coco/calibration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from lead_net.models import build_lead_net
from lead_net.data import build_coco_dataset
from lead_net.engine import Evaluator
from lead_net.utils import load_config, get_nested, ExperimentManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net INT8 量化精度对比")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--fp32-weights", required=True, type=str,
                   help="FP32 checkpoint 路径")
    p.add_argument("--int8-model", type=str, default=None,
                   help="INT8 TFLite 模型路径（留空则仅评估 FP32）")
    p.add_argument("--calibration-dir", type=str, default=None,
                   help="校准集目录（记录用）")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", type=str,
                   default="outputs/experiments/quantization_comparison.csv")
    return p.parse_args()


def eval_fp32(cfg, weights_path, device, max_samples) -> dict:
    """评估 FP32 模型，返回 per-class AP。"""
    model = build_lead_net(cfg)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    ds = build_coco_dataset(cfg, split="val")
    if max_samples:
        ds.ids = ds.ids[:max_samples]

    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False,
                                          collate_fn=lambda x: x, num_workers=0)
    evaluator = Evaluator(model, loader, cfg, device)
    return evaluator.evaluate()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = get_nested(cfg, "experiment.tag", "model")
    class_map = cfg.get("class_map", {})

    t0 = time.time()

    # FP32 baseline
    print("[eval] FP32 model...")
    fp32_result = eval_fp32(cfg, args.fp32_weights, device, args.max_samples)
    fp32_per_class = fp32_result.get("per_class", [])
    fp32_time = time.time() - t0

    # INT8 (if available)
    int8_per_class = []
    int8_time = 0.0
    if args.int8_model:
        t1 = time.time()
        print("[eval] INT8 model...")
        # TODO M5: 实际 TFLite 推理调用
        # import tensorflow as tf
        # interpreter = tf.lite.Interpreter(model_path=args.int8_model)
        # ... run inference on val set ...
        # int8_per_class = ...
        print("[warn] INT8 evaluation not yet implemented (needs M5 TFLite model)")
        int8_time = time.time() - t1

    # ---- 对比输出 (to variant quantization/ subdir) ----
    variant = "lca" if "lca" in tag.lower() else "baseline"
    mgr = ExperimentManager.for_test("outputs/experiments", variant)
    csv_path = mgr.quantization_dir / f"{tag}_int8_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("class_id,class_name,fp32_AP50,fp32_AP_coco,int8_AP50,int8_AP_coco,d_AP50,d_AP_coco\n")
        for fp32_cls in fp32_per_class:
            cls_id = fp32_cls["class_id"]
            name = class_map.get(str(cls_id), f"cls_{cls_id}")
            fp50 = fp32_cls.get("AP@0.5", 0)
            fpcoco = fp32_cls.get("AP@0.5:0.95", 0)

            # 找对应 INT8 条目
            int8_match = next((c for c in int8_per_class if c.get("class_id") == cls_id), {})
            i50 = int8_match.get("AP@0.5", None)
            icoco = int8_match.get("AP@0.5:0.95", None)

            d50 = (fp50 - i50) if (fp50 is not None and i50 is not None) else ""
            dcoco = (fpcoco - icoco) if (fpcoco is not None and icoco is not None) else ""

            f.write(f"{cls_id},{name},{fp50:.4f},{fpcoco:.4f},{i50},{icoco},{d50},{dcoco}\n")

        # 元数据注释行
        f.write(f"# calibration_dir,{args.calibration_dir or 'N/A'}\n")
        f.write(f"# calibration_size,{args.max_samples or cfg.get('data',{}).get('val_split','val2017')}\n")
        f.write(f"# fp32_eval_time_s,{fp32_time:.1f}\n")
        f.write(f"# int8_eval_time_s,{int8_time:.1f}\n")

    # 摘要
    fp32_map50 = fp32_result.get("mAP@0.5", 0)
    print(f"\n=== Quantization Comparison ===")
    print(f"  FP32 mAP@0.5:      {fp32_map50:.4f}")
    if int8_per_class:
        int8_map50 = np.mean([c.get("AP@0.5", 0) for c in int8_per_class])
        print(f"  INT8 mAP@0.5:      {int8_map50:.4f}")
        print(f"  d_mAP@0.5:         {fp32_map50 - int8_map50:.4f}")
    print(f"  FP32 eval time:    {fp32_time:.1f}s")
    print(f"  INT8 eval time:    {int8_time:.1f}s")
    print(f"  CSV → {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
