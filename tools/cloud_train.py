"""cloud_train.py — 云端 RTX 5090 (32GB) 压榨训练脚本。

职责（单一）：
    在云端 RTX 5090 32GB + 25 vCPU + 90GB RAM 上，压榨 GPU 跑 4 个变体完整训练：
      - baseline: YOLO11n 7类 (Stage 0 教师基准)
      - lca_r16:  YOLO11n + LCA(Neck, r=16) (主实验)
      - lca_r8:   YOLO11n + LCA(Neck, r=8)  (reduction 消融)
      - lca_r32:  YOLO11n + LCA(Neck, r=32) (reduction 消融)
    每个变体训练完自动评估，输出对比报告。

云端压榨参数（RTX 5090 32GB）：
    - batch=256: 512 实测触发 TaskAlignedAssigner OOM 回退 CPU（2026-07-17，
      assigner 中间张量 (bs, n_max_boxes, h*w) 随 batch 线性放大，峰值冲破 32GB）；
      仍出现该警告再降 --batch 192 或 128
    - workers=24: 25 vCPU 留 1 个给系统
    - imgsz=416: 比本地 320 更高分辨率，小目标精度更好
    - cache="ram": 13576 张图约 3GB，90GB RAM 充裕
    - amp=True: 混合精度
    - epochs=180: 完整训练（云端算力够）
    - cos_lr + lr0=0.01 + warmup

用法（云端 Linux）：
    PYTHONPATH=. python tools/cloud_train.py --variants baseline lca_r16 lca_r8 lca_r32
    PYTHONPATH=. python tools/cloud_train.py --variants baseline lca_r16  # 只跑主对比

产出：
    outputs/cloud/runs/<variant>/weights/best.pt  # 最优权重
    outputs/cloud/runs/<variant>/results.csv      # 训练曲线
    outputs/cloud/report/cloud_train_summary.json # 对比报告
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# 注册 LCA（导入即注册）
from lead_net.models.yolo import lca_adapter  # noqa: F401
from lead_net.models.yolo.data_adapter import write_data_yaml

import torch
from ultralytics import YOLO


YAMLS = {
    "baseline": "lead_net/models/yolo/yamls/yolo11n_lead.yaml",
    "lca_r16": "lead_net/models/yolo/yamls/yolo11n_lca_neck_r16.yaml",
    "lca_r8": "lead_net/models/yolo/yamls/yolo11n_lca_neck_r8.yaml",
    "lca_r32": "lead_net/models/yolo/yamls/yolo11n_lca_neck_r32.yaml",
}
LCA_INSERT_INDEX = 17  # LCA 插入位置（用于权重重映射）
DATA_YAML = "configs/lead_subset_ultralytics.yaml"
REPORT_DIR = Path("outputs/cloud/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def train_one_variant(variant_name: str, epochs: int, imgsz: int,
                      batch: int, workers: int, data_yaml: str) -> dict:
    """训练一个变体并评估。"""
    print(f"\n{'='*70}")
    print(f"CLOUD TRAIN: {variant_name} (epochs={epochs}, imgsz={imgsz}, batch={batch})")
    print(f"{'='*70}")

    t0 = time.time()
    yaml_path = YAMLS[variant_name]

    # 权重加载：LCA 变体用重映射，baseline 用原生
    if "lca" in variant_name.lower():
        from lead_net.models.yolo.weight_remapper import load_pretrained_with_remapping
        model = load_pretrained_with_remapping(
            yaml_path, "yolo11n.pt", lca_insert_index=LCA_INSERT_INDEX, verbose=True
        )
    else:
        model = YOLO(yaml_path).load("yolo11n.pt")

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"params: {n_params/1e6:.4f} M")

    # cudnn 自动调优
    torch.backends.cudnn.benchmark = True

    # 云端压榨训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=workers,
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        patience=30,  # 完整训练允许早停
        verbose=True,
        project="outputs/cloud/runs",
        name=variant_name,
        exist_ok=True,
        save=True,
        save_period=10,  # 每 10 epoch 存一次
        plots=True,
        cache="ram",
        amp=True,
        close_mosaic=10,
    )
    train_time = time.time() - t0

    # 评估
    print(f"\n[eval] {variant_name} on val full set...")
    metrics = model.val(data=data_yaml, imgsz=imgsz, split="val", verbose=True)

    return {
        "variant": variant_name,
        "yaml": yaml_path,
        "params": n_params,
        "params_M": round(n_params / 1e6, 4),
        "train_time_s": round(train_time, 1),
        "train_time_h": round(train_time / 3600, 2),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "metrics": {
            "mAP@0.5": float(metrics.box.map50),
            "mAP@0.5:0.95": float(metrics.box.map),
            "mAP@0.75": float(metrics.box.map75),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "per_class_mAP@0.5": [float(x) for x in metrics.box.maps],
        },
        "best_weights": f"outputs/cloud/runs/{variant_name}/weights/best.pt",
        "last_weights": f"outputs/cloud/runs/{variant_name}/weights/last.pt",
    }


def main():
    parser = argparse.ArgumentParser(description="云端 RTX 5090 压榨训练")
    parser.add_argument("--variants", nargs="+",
                        default=["baseline", "lca_r16", "lca_r8", "lca_r32"],
                        choices=list(YAMLS.keys()))
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--smoke", action="store_true", help="1 epoch 冒烟")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.batch = 64
        args.workers = 4

    # 确保 data.yaml
    write_data_yaml("data/lead_subset", DATA_YAML)

    print(f"Cloud training config:")
    print(f"  variants: {args.variants}")
    print(f"  epochs: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}, workers: {args.workers}")
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    results = {}
    for v in args.variants:
        results[v] = train_one_variant(v, args.epochs, args.imgsz, args.batch, args.workers, DATA_YAML)

    # 对比汇总
    print(f"\n{'='*70}")
    print("CLOUD TRAIN SUMMARY")
    print(f"{'='*70}")
    print(f"{'variant':<12} {'params(M)':<12} {'mAP@0.5':<10} {'mAP@0.5:0.95':<14} "
          f"{'recall':<10} {'time(h)':<10}")
    for v, r in results.items():
        m = r["metrics"]
        print(f"{v:<12} {r['params_M']:<12} {m['mAP@0.5']:<10.4f} "
              f"{m['mAP@0.5:0.95']:<14.4f} {m['recall']:<10.4f} {r['train_time_h']:<10}")

    # LCA 增益
    if "baseline" in results:
        b = results["baseline"]["metrics"]["mAP@0.5"]
        for v in results:
            if v != "baseline":
                diff = results[v]["metrics"]["mAP@0.5"] - b
                print(f"LCA gain ({v} - baseline): {diff:+.4f} ({'✅有效' if diff > 0 else '❌无效'})")

    out_path = REPORT_DIR / "cloud_train_summary.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] report saved to {out_path}")
    print(f"权重在 outputs/cloud/runs/<variant>/weights/best.pt")


if __name__ == "__main__":
    main()
