#!/usr/bin/env python
"""LR Range Test CLI —— 自动搜索最优初始学习率。

基于 Leslie Smith (2015/2018) "Cyclical Learning Rates for Training Neural Networks"

用法:
    python tools/lr_range_test.py --config configs/train_lca.yaml
    python tools/lr_range_test.py --config configs/train_lca.yaml --steps 500 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lead_net.utils.config import load_config
from lead_net.models.lead_net import build_lead_net
from lead_net.models.loss import MultiBoxLoss
from lead_net.data import build_dataloader, collate_fn
from lead_net.engine.llrd import build_llrd_param_groups
from lead_net.engine.lr_range_test import LRRangeTest, run_lr_range_test


def parse_args():
    p = argparse.ArgumentParser(description="LR Range Test — automatic LR finder")
    p.add_argument("--config", required=True, help="训练配置文件")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--steps", type=int, default=500, help="测试步数")
    p.add_argument("--start-lr", type=float, default=1e-7)
    p.add_argument("--end-lr", type=float, default=1.0)
    p.add_argument("--safety", type=float, default=10.0, help="安全除数 (Leslie Smith 建议 10)")
    p.add_argument("--output", default="outputs/lr_range_test", help="输出目录")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  LR Range Test")
    print(f"  Config: {args.config}")
    print(f"  Steps:  {args.steps}")
    print(f"  Range:  {args.start_lr:.1e} → {args.end_lr:.1e}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # ─── DataLoader ───
    loader = build_dataloader(cfg, split="train",
                              num_workers=min(4, cfg.get("training", {}).get("batch_size", 8) // 2),
                              shuffle=True)
    gpu_proc = getattr(loader, "gpu_processor", None)

    # ─── 模型 + 损失 ───
    model = build_lead_net(cfg)
    model.to(device)
    model.train()
    criterion = MultiBoxLoss(
        num_classes=cfg.get("num_classes", 7) + 1,
        input_size=cfg.get("data", {}).get("input_size", 320),
    )

    # ─── 优化器（统一 LR 组用于测试） ───
    param_groups = build_llrd_param_groups(model, cfg, freeze_backbone=False)
    # 统一所有组的初始 LR
    for g in param_groups:
        g["lr"] = 0.001
        g["initial_lr"] = 0.001
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.SGD(
        param_groups,
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 5e-4),
        nesterov=opt_cfg.get("nesterov", True),
    )

    # ─── 运行 LR Range Test ───
    print(f"\n  Running LR Range Test ({args.steps} steps)...\n")
    result = run_lr_range_test(
        model=model,
        train_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        steps=args.steps,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        safety_factor=args.safety,
    )

    # ─── 输出 ───
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"  {'='*60}")

    if result.get("error"):
        print(f"  Error: {result['error']}")
        sys.exit(1)

    print(f"  Recommended LR (head):       {result['optimal_lr']:.2e}")
    print(f"  LR at min loss:              {result['lr_at_min_loss']:.2e}")
    print(f"  LR at steepest descent:      {result['lr_at_steepest_descent']:.2e}")
    print(f"  Min loss:                    {result['min_loss']:.4f}")
    print(f"  Steps tested:                {result['steps_tested']}")
    print(f"  Safety factor:               {result['safety_factor']}")
    if "elapsed_seconds" in result:
        print(f"  Time:                        {result['elapsed_seconds']:.1f}s")

    # 为 LLRD 各组生成建议
    opt_lr = result["optimal_lr"]
    print(f"\n  Suggested LLRD config:")
    print(f"    head:           {opt_lr:.2e}")
    print(f"    lca:            {opt_lr * 0.8:.2e}")
    print(f"    backbone_last:  {opt_lr * 0.3:.2e}")
    print(f"    backbone_middle:{opt_lr * 0.1:.2e}")
    print(f"    backbone_first: {opt_lr * 0.03:.2e}")

    # ─── 保存 ───
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "device": str(device),
        "start_lr": args.start_lr,
        "end_lr": args.end_lr,
        "steps": args.steps,
        "safety_factor": args.safety,
        **{k: v for k, v in result.items() if k != "history" and k != "smoothed_history"},
        "suggested_llrd": {
            "head": opt_lr,
            "lca": opt_lr * 0.8,
            "backbone_last": opt_lr * 0.3,
            "backbone_middle": opt_lr * 0.1,
            "backbone_first": opt_lr * 0.03,
        },
        "history": result.get("history", []),
    }

    report_path = out_dir / "lr_range_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
