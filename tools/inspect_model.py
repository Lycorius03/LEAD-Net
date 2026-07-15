"""模型开销分析工具 —— 按模块拆分参数量 + 文件体积。

用途（对应 RQ2）：
    - 分别统计 Backbone / LCA / SSD-Lite Head 的参数量
    - 记录模型文件大小（FP32 .pth）
    - 区分 Baseline 和 +LCA 版本对比

用法：
    python tools/inspect_model.py --config configs/baseline_ssd.yaml
    python tools/inspect_model.py --config configs/lca_ssd.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lead_net.utils import load_config, get_nested, ExperimentManager
from lead_net.models import build_lead_net


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net 模型开销分析")
    p.add_argument("--config", required=True, type=str,
                   help="配置文件路径")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="checkpoint 路径（可选，用于测文件大小）")
    return p.parse_args()


def count_params(module, name: str = "total") -> dict[str, int]:
    """递归统计模块参数。"""
    result = {}
    n = sum(p.numel() for p in module.parameters())
    result[name] = n

    # 按子模块拆分
    for child_name, child in module.named_children():
        child_n = sum(p.numel() for p in child.parameters())
        if child_n > 0:
            result[f"{name}/{child_name}"] = child_n

    return result


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    tag = get_nested(cfg, "experiment.tag", "model")
    use_lca = get_nested(cfg, "model.lca.enabled", False)

    print(f"=== Model Profile: {tag} ===")
    print(f"LCA: {'enabled' if use_lca else 'disabled'}")
    print()

    # 构建模型
    model = build_lead_net(cfg)

    # ---- 参数量统计 ----
    backbone = model.backbone
    head = model.head

    # Backbone params (分：features + proj_s16 + proj_s32 + extra)
    backbone_total = sum(p.numel() for p in backbone.parameters())
    features_params = sum(p.numel() for p in backbone.features.parameters())
    proj_s16_params = sum(p.numel() for p in backbone.proj_s16.parameters())
    proj_s32_params = sum(p.numel() for p in backbone.proj_s32.parameters())
    extra_params = sum(p.numel() for p in backbone.extra.parameters())

    # LCA (若启用)
    lca_params = 0
    if use_lca and backbone.lca is not None:
        lca_params = sum(p.numel() for p in backbone.lca.parameters())

    # SSD Head
    head_params = sum(p.numel() for p in head.parameters())

    total_params = sum(p.numel() for p in model.parameters())

    # ---- 输出 ----
    print("--- Parameter Count (by module) ---")
    print(f"  Backbone (total):        {backbone_total:>10,}")
    print(f"    - MobileNetV3 features: {features_params:>10,}")
    print(f"    - proj_s16 (48→256):    {proj_s16_params:>10,}")
    print(f"    - proj_s32 (576→256):   {proj_s32_params:>10,}")
    print(f"    - extra (stride 64):    {extra_params:>10,}")
    if use_lca:
        print(f"  LCA Attention:            {lca_params:>10,}  ({lca_params/backbone_total*100:.2f}% of backbone)")
    print(f"  SSD-Lite Head:            {head_params:>10,}")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL:                    {total_params:>10,}")
    print()

    # ---- 输出到实验目录 ----
    variant = "lca" if use_lca else "baseline"
    mgr = ExperimentManager.for_test("outputs/experiments", variant)
    csv_path = mgr.run_dir / "model_profile.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("module,params\n")
        f.write(f"backbone_features,{features_params}\n")
        f.write(f"backbone_proj_s16,{proj_s16_params}\n")
        f.write(f"backbone_proj_s32,{proj_s32_params}\n")
        f.write(f"backbone_extra,{extra_params}\n")
        f.write(f"lca,{lca_params}\n")
        f.write(f"head,{head_params}\n")
        f.write(f"total,{total_params}\n")
    print(f"[info] CSV saved to {csv_path}")

    # ---- 文件大小 ----
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt.exists():
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"\n--- Model File Size ---")
            print(f"  {ckpt.name}: {size_mb:.1f} MB (FP32)")
        else:
            print(f"[warn] checkpoint not found: {ckpt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
