#!/usr/bin/env python
"""快速 mAP 验证测试 —— 100-200张图片短训练 + mAP评估。

目的：
    验证 mAP 修复是否有效。在 150 张图片上训练 15 epoch，
    然后运行 COCO mAP 评估，确认 mAP@0.5 > 0。

用法:
    python tools/quick_map_test.py --config configs/train_baseline.yaml --samples 150 --epochs 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lead_net.utils.config import load_config
from lead_net.models.lead_net import build_lead_net
from lead_net.models.loss import MultiBoxLoss
from lead_net.data import build_dataloader
from lead_net.engine.llrd import build_llrd_param_groups
from lead_net.engine.scheduler import build_scheduler_from_total_iters
from lead_net.engine.evaluator import Evaluator
from lead_net.engine.checkpoint import CheckpointManager
from lead_net.engine.ema import ModelEMA


def parse_args():
    p = argparse.ArgumentParser(description="Quick mAP verification test")
    p.add_argument("--config", required=True, help="训练配置文件")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--samples", type=int, default=150, help="训练图片数")
    p.add_argument("--epochs", type=int, default=15, help="训练 epoch 数")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001, help="统一LR")
    p.add_argument("--eval-samples", type=int, default=50, help="评估图片数")
    p.add_argument("--score-threshold", type=float, default=0.01,
                   help="评估置信度阈值（低阈值用于诊断）")
    p.add_argument("--output", default="outputs/quick_map_test", help="输出目录")
    return p.parse_args()


def limit_dataset(dataset, n: int):
    """取前 n 个样本。"""
    return Subset(dataset, list(range(min(n, len(dataset)))))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Quick mAP Test")
    print(f"  Train samples: {args.samples} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"  Eval samples: {args.eval_samples} | Score thr: {args.score_threshold}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # ─── 训练数据（子集） ───
    full_train_loader = build_dataloader(
        cfg, split="train", batch_size=args.batch_size,
        num_workers=0, shuffle=True,
    )
    train_dataset = limit_dataset(full_train_loader.dataset, args.samples)
    # 重新构建 DataLoader（使用 subset）
    from torch.utils.data import DataLoader
    from lead_net.data.dataloader import collate_fn
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    steps_per_epoch = len(train_loader)
    total_iters = args.epochs * steps_per_epoch
    print(f"  Steps/epoch: {steps_per_epoch} | Total iters: {total_iters}")

    # ─── 验证数据（子集） ───
    full_val_loader = build_dataloader(
        cfg, split="val", batch_size=args.batch_size,
        num_workers=0, shuffle=False,
    )
    val_dataset = limit_dataset(full_val_loader.dataset, args.eval_samples)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    # ─── 模型 + 损失 ───
    model = build_lead_net(cfg)
    model.to(device)
    model.train()
    criterion = MultiBoxLoss(cfg)

    # ─── EMA ───
    ema = ModelEMA(model, decay=0.9998)

    # ─── 优化器 + 调度器 ───
    param_groups = build_llrd_param_groups(model, cfg, freeze_backbone=False)
    if args.lr > 0:
        for g in param_groups:
            g["lr"] = args.lr
            g["initial_lr"] = args.lr
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.SGD(
        param_groups,
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 5e-4),
        nesterov=opt_cfg.get("nesterov", True),
    )
    scheduler = build_scheduler_from_total_iters(
        optimizer,
        warmup_iters=min(50, total_iters // 10),
        total_iters=total_iters,
        warmup_start_factor=0.05,
        eta_min_factor=0.0,
    )

    # ─── AMP ───
    use_amp = cfg.get("training", {}).get("amp", False) and device.type == "cuda"
    scaler_init = cfg.get("training", {}).get("grad_scaler_init_scale", 2048.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=scaler_init) if use_amp else None

    # ─── 训练循环 ───
    t_start = time.time()
    history = []

    for epoch in range(args.epochs):
        epoch_cls = epoch_loc = epoch_loss = 0.0
        n = 0
        t_e_start = time.time()

        for batch in train_loader:
            images = batch["image"].to(device)
            gt_boxes = batch["boxes"]
            gt_labels = batch["labels"]

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    cls_pred, loc_pred = model(images)
                    default_boxes = model.head.all_default_boxes(device)
                    cls_loss, loc_loss = criterion(cls_pred, loc_pred, default_boxes,
                                                    gt_boxes, gt_labels)
                    loss = cls_loss + loc_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                cls_pred, loc_pred = model(images)
                default_boxes = model.head.all_default_boxes(device)
                cls_loss, loc_loss = criterion(cls_pred, loc_pred, default_boxes,
                                                gt_boxes, gt_labels)
                loss = cls_loss + loc_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            scheduler.step()
            ema.update()

            if not (torch.isnan(loss) or torch.isinf(loss)):
                epoch_cls += cls_loss.item()
                epoch_loc += loc_loss.item()
                epoch_loss += loss.item()
                n += 1

        avg_cls = epoch_cls / max(n, 1)
        avg_loc = epoch_loc / max(n, 1)
        avg_loss = epoch_loss / max(n, 1)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_e_start

        history.append({
            "epoch": epoch + 1,
            "cls": round(avg_cls, 4),
            "loc": round(avg_loc, 4),
            "loss": round(avg_loss, 4),
            "lr": lr,
            "time_s": round(elapsed, 1),
        })

        print(f"  epoch {epoch+1:3d}/{args.epochs}  "
              f"cls={avg_cls:.4f}  loc={avg_loc:.4f}  loss={avg_loss:.4f}  "
              f"lr={lr:.2e}  {elapsed:.0f}s")

    train_time = time.time() - t_start

    # ─── mAP 评估 ───
    print(f"\n{'='*60}")
    print(f"  Running mAP evaluation (score_thr={args.score_threshold})...")

    # 使用 EMA 权重评估
    ema.apply()
    model.eval()

    # 临时覆盖 eval 配置
    eval_cfg_backup = cfg.get("eval", {}).copy()
    cfg.setdefault("eval", {})["score_threshold"] = args.score_threshold

    evaluator = Evaluator(model=model, val_loader=val_loader, cfg=cfg, device=device)
    metrics = evaluator.evaluate()

    # 恢复配置
    if eval_cfg_backup:
        cfg["eval"] = eval_cfg_backup

    ema.restore()

    mAP50 = metrics.get("mAP@0.5", 0.0)
    mAP50_95 = metrics.get("mAP@0.5:0.95", 0.0)
    mAP75 = metrics.get("mAP@0.75", 0.0)

    print(f"  mAP@0.5:      {mAP50:.4f}")
    print(f"  mAP@0.5:0.95: {mAP50_95:.4f}")
    print(f"  mAP@0.75:     {mAP75:.4f}")

    if metrics.get("per_class"):
        print(f"  Per-class AP@0.5:")
        for cls_info in metrics["per_class"]:
            print(f"    {cls_info['class_name']:15s}: {cls_info['AP@0.5']:.4f}")

    # ─── 判断 ───
    print(f"\n{'='*60}")
    print(f"  Results")

    initial_loss = history[0]["loss"]
    final_loss = history[-1]["loss"]
    loss_decreased = final_loss < initial_loss * 0.8

    print(f"  Train loss: {initial_loss:.4f} -> {final_loss:.4f}")
    print(f"  Train time: {train_time:.0f}s ({train_time/60:.1f}min)")
    print(f"  mAP@0.5: {mAP50:.4f}")

    if not loss_decreased:
        print("\n  [FAIL] Loss did not decrease sufficiently")
        verdict = "FAIL"
    elif mAP50 > 0.001:
        print(f"\n  [PASS] mAP@0.5 = {mAP50:.4f} > 0.001! Fix working!")
        verdict = "PASS"
    elif mAP50 > 0.0:
        print(f"\n  [MARGINAL] mAP@0.5 = {mAP50:.6f} > 0 but very low. "
              f"Fix is working, needs more training.")
        verdict = "MARGINAL"
    elif mAP50 == 0.0 or mAP50 < 1e-8:
        print("\n  [INCONCLUSIVE] mAP@0.5 ≈ 0.0")
        print("  Possible causes:")
        print("    1. Too few training samples/epochs for meaningful detection")
        print("    2. Score threshold still too high for early training")
        print("    3. Further pipeline debugging needed")
        print("  Try: --score-threshold 0.001 --epochs 30 --samples 500")
        verdict = "INCONCLUSIVE"
    else:
        verdict = "UNKNOWN"

    # ─── 保存报告 ───
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "device": str(device),
        "train_samples": args.samples,
        "eval_samples": args.eval_samples,
        "epochs": args.epochs,
        "lr": args.lr,
        "score_threshold": args.score_threshold,
        "verdict": verdict,
        "train_time_s": round(train_time, 1),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "mAP@0.5": mAP50,
        "mAP@0.5:0.95": mAP50_95,
        "mAP@0.75": mAP75,
        "per_class": metrics.get("per_class", []),
        "history": history,
    }

    report_path = out_dir / "quick_map_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to: {report_path}")

    sys.exit(0 if verdict == "PASS" else (1 if verdict == "FAIL" else 0))


if __name__ == "__main__":
    main()
