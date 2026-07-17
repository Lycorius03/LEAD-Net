"""stage1c_lca_gain_train.py — Stage 1Cc LCA 增益验证短训练。

职责（单一）：
    跑 4 组短训练对比，验证 LCA 在 YOLO11n 教师上是否真正提升困难场景：
      A0: YOLO11n Baseline (无 LCA)
      A1: YOLO11n + LCA(Neck, r=16)
    每组在 val 全集 + 3 困难子集（小目标/遮挡/红色背景）上评估 mAP@0.5。
    同时对比 reduction r=8/32（Stage 2 消融预热）。

不负责：
    - 完整 180 epoch 训练（阶段 3-5）
    - 学生模型蒸馏（阶段 6）
    - 部署导出（阶段 7）

用法：
    python tools/stage1c_lca_gain_train.py --epochs 15 --smoke
    python tools/stage1c_lca_gain_train.py --epochs 15
    产出：outputs/stage1c/report/lca_gain.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# 注册 LCA 到 ultralytics（导入即注册）
from lead_net.models.yolo import lca_adapter  # noqa: F401
from lead_net.models.yolo.data_adapter import write_data_yaml

from ultralytics import YOLO


YAML_BASELINE = "lead_net/models/yolo/yamls/yolo11n_lead.yaml"
YAML_LCA_R16 = "lead_net/models/yolo/yamls/yolo11n_lca_neck_r16.yaml"
DATA_YAML = "configs/lead_subset_ultralytics.yaml"
SUBSETS_DIR = Path("outputs/stage1c/subsets")
REPORT_DIR = Path("outputs/stage1c/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_on_subsets(model: YOLO, data_yaml: str, subsets_dir: Path) -> dict:
    """在 val 全集 + 3 困难子集上评估。

    ultralytics val() 不直接支持子集评估，这里用变通：
        - 全集：标准 val()
        - 子集：构造临时 data.yaml 指向子集图片列表
    简化：直接用 val() 在全集上跑，子集指标靠后处理筛选（需 predictions）。
    本轮简化：只测全集 mAP，子集留 Stage 1Cc 后续。
    """
    metrics = model.val(data=data_yaml, imgsz=320, split="val", verbose=False)
    return {
        "mAP@0.5": float(metrics.box.map50),
        "mAP@0.5:0.95": float(metrics.box.map),
        "mAP@0.75": float(metrics.box.map75),
        "per_class_mAP@0.5": [float(x) for x in metrics.box.maps],
    }


def train_one_variant(yaml_path: str, variant_name: str, epochs: int,
                      data_yaml: str, smoke: bool = False) -> dict:
    """训练一个变体并评估。"""
    print(f"\n{'='*70}")
    print(f"Training {variant_name} (epochs={epochs}, smoke={smoke})")
    print(f"{'='*70}")

    t0 = time.time()
    # Baseline 用原生 .load()，LCA 变体用 weight_remapper（补偿索引偏移）
    if "lca" in variant_name.lower():
        from lead_net.models.yolo.weight_remapper import load_pretrained_with_remapping
        model = load_pretrained_with_remapping(yaml_path, "yolo11n.pt", lca_insert_index=17, verbose=True)
    else:
        model = YOLO(yaml_path).load("yolo11n.pt")
    n_params = sum(p.numel() for p in model.model.parameters())

    # GPU 压榨参数（RTX 5060 Laptop 8.55GB, Windows）
    # - batch=128: YOLO11n @320 显存约 5GB，留余量给数据 pipeline
    # - workers=4: Windows 多进程 DataLoader 兼容性，Linux 可调 16
    # - cache="ram": 13576 张图预加载到 RAM（约 3GB），避免磁盘 IO 瓶颈
    # - amp=True: 混合精度
    # - 加 cudnn.benchmark=True 自动调优（通过 env 设）
    import torch
    torch.backends.cudnn.benchmark = True

    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=320,
        batch=128,
        device=0,
        workers=4,  # Windows 兼容，不是 16
        cos_lr=True,
        lr0=0.01,
        patience=0,
        verbose=True,
        project="outputs/stage1c/runs",
        name=variant_name,
        exist_ok=True,
        save=True,
        plots=False,
        cache="ram",  # 预加载到 RAM
        amp=True,
    )
    if smoke:
        train_kwargs.update(epochs=1, batch=64, workers=2, cache=False)

    results = model.train(**train_kwargs)
    train_time = time.time() - t0

    # 评估
    print(f"\n[eval] {variant_name} on val full set...")
    metrics = evaluate_on_subsets(model, data_yaml, SUBSETS_DIR)

    return {
        "variant": variant_name,
        "yaml": yaml_path,
        "params": n_params,
        "params_M": round(n_params / 1e6, 4),
        "train_time_s": round(train_time, 1),
        "epochs": epochs,
        "smoke": smoke,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--smoke", action="store_true", help="1 epoch 冒烟测试")
    parser.add_argument("--variants", nargs="+", default=["baseline", "lca_r16"],
                        choices=["baseline", "lca_r16", "lca_r8", "lca_r32"])
    args = parser.parse_args()

    # 确保 data.yaml 存在
    write_data_yaml("data/lead_subset", DATA_YAML)

    results = {}
    variant_map = {
        "baseline": YAML_BASELINE,
        "lca_r16": YAML_LCA_R16,
    }

    for v in args.variants:
        yaml_path = variant_map.get(v)
        if yaml_path is None:
            print(f"[skip] {v} 暂未实现（Stage 2 消融）")
            continue
        results[v] = train_one_variant(yaml_path, v, args.epochs, DATA_YAML, args.smoke)

    # 对比汇总
    print(f"\n{'='*70}")
    print("Stage 1Cc LCA Gain Summary")
    print(f"{'='*70}")
    print(f"{'variant':<12} {'params(M)':<12} {'mAP@0.5':<10} {'mAP@0.5:0.95':<14} {'time(s)':<10}")
    for v, r in results.items():
        m = r["metrics"]
        print(f"{v:<12} {r['params_M']:<12} {m['mAP@0.5']:<10.4f} "
              f"{m['mAP@0.5:0.95']:<14.4f} {r['train_time_s']:<10}")

    # LCA 增益差值
    if "baseline" in results and "lca_r16" in results:
        b = results["baseline"]["metrics"]["mAP@0.5"]
        l = results["lca_r16"]["metrics"]["mAP@0.5"]
        diff = l - b
        print(f"\nLCA gain (mAP@0.5): {diff:+.4f} ({'✅ 有效' if diff > 0 else '❌ 无效/装饰'})")

    out_path = REPORT_DIR / "lca_gain.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] report saved to {out_path}")


if __name__ == "__main__":
    main()
