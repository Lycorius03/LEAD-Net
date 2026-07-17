"""smoke_verify_models.py — 小样本冒烟验证两个模型代码能跑通。

职责（单一）：
    用 100 张 train / 50 张 val 小样本，跑 2 epoch，验证：
      - baseline (yolo11n_lead.yaml) 能加载预训练 + 训练 + 评估
      - lca_r16 (yolo11n_lca_neck_r16.yaml) 能加载预训练(重映射) + 训练 + 评估
    不追求精度，只验证代码管线无 bug。

用法：
    python tools/smoke_verify_models.py
"""
from __future__ import annotations

import time
from pathlib import Path

from lead_net.models.yolo import lca_adapter  # noqa: F401
from ultralytics import YOLO


SMOKE_DATA_YAML = "configs/lead_subset_smoke.yaml"
YAML_BASELINE = "lead_net/models/yolo/yamls/yolo11n_lead.yaml"
YAML_LCA_R16 = "lead_net/models/yolo/yamls/yolo11n_lca_neck_r16.yaml"


SMOKE_DATA_CONTENT = """\
# Smoke test dataset (100 train / 50 val)
path: {root}
train: images/train
val: images/val
names:
  0: person
  1: bicycle
  2: car
  3: backpack
  4: suitcase
  5: chair
  6: bottle
"""


def write_smoke_yaml() -> str:
    root = Path("data/lead_subset_smoke").resolve().as_posix()
    Path(SMOKE_DATA_YAML).write_text(SMOKE_DATA_CONTENT.format(root=root), encoding="utf-8")
    return SMOKE_DATA_YAML


def smoke_one(yaml_path: str, name: str, data_yaml: str) -> dict:
    print(f"\n{'='*60}")
    print(f"SMOKE: {name} ({yaml_path})")
    print(f"{'='*60}")
    t0 = time.time()

    if "lca" in name.lower():
        from lead_net.models.yolo.weight_remapper import load_pretrained_with_remapping
        model = load_pretrained_with_remapping(yaml_path, "yolo11n.pt", lca_insert_index=17, verbose=True)
    else:
        model = YOLO(yaml_path).load("yolo11n.pt")

    n_params = sum(p.numel() for p in model.model.parameters())

    results = model.train(
        data=data_yaml,
        epochs=2,
        imgsz=320,
        batch=16,
        device=0,
        workers=2,
        cos_lr=True,
        lr0=0.01,
        patience=0,
        verbose=True,
        project="outputs/stage1c/smoke",
        name=name,
        exist_ok=True,
        save=True,
        plots=False,
        cache=False,
        amp=True,
    )
    dt = time.time() - t0

    # 评估
    metrics = model.val(data=data_yaml, imgsz=320, split="val", verbose=False)
    return {
        "name": name,
        "params_M": round(n_params / 1e6, 4),
        "time_s": round(dt, 1),
        "mAP@0.5": float(metrics.box.map50),
        "mAP@0.5:0.95": float(metrics.box.map),
        "status": "PASS" if metrics.box.map50 >= 0 else "FAIL",
    }


def main():
    data_yaml = write_smoke_yaml()
    print(f"smoke data.yaml: {data_yaml}")

    results = []
    for name, yaml in [("baseline", YAML_BASELINE), ("lca_r16", YAML_LCA_R16)]:
        try:
            r = smoke_one(yaml, name, data_yaml)
            results.append(r)
            print(f"  {name}: {r['status']} mAP@0.5={r['mAP@0.5']:.4f} params={r['params_M']}M time={r['time_s']}s")
        except Exception as e:
            results.append({"name": name, "status": "ERROR", "error": str(e)})
            print(f"  {name}: ERROR {e}")

    print(f"\n{'='*60}")
    print("SMOKE SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['name']:<12} {r['status']:<8} mAP@0.5={r.get('mAP@0.5','N/A')}")
    print(f"\n所有模型代码管线验证: {'✅ 全部 PASS' if all(r['status']=='PASS' for r in results) else '❌ 有失败'}")


if __name__ == "__main__":
    main()
