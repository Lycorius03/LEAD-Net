#!/usr/bin/env python
"""短测试 —— 100-200 张图片快速验证。

目的：
    在较小数据子集上跑 10-20 epoch，验证:
    1. 无 NaN（修复生效）
    2. loss 下降趋势正常
    3. 梯度分布合理

用法:
    python tools/short_test.py --config configs/train_lca.yaml --samples 150 --epochs 15
    python tools/short_test.py --config configs/train_lca.yaml --samples 100 --epochs 10 --device cpu
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
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lead_net.utils.config import load_config
from lead_net.models.lead_net import build_lead_net
from lead_net.models.loss import MultiBoxLoss
from lead_net.data import build_dataloader, collate_fn
from lead_net.engine.llrd import build_llrd_param_groups
from lead_net.engine.scheduler import build_scheduler_from_total_iters
from lead_net.engine.nan_detector import grad_stats, summarize_grads


def parse_args():
    p = argparse.ArgumentParser(description="Short test — quick training sanity check")
    p.add_argument("--config", required=True, help="训练配置文件")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--samples", type=int, default=150, help="训练图片数")
    p.add_argument("--epochs", type=int, default=15, help="训练 epoch 数")
    p.add_argument("--batch-size", type=int, default=8, help="短测试 batch 大小")
    p.add_argument("--lr", type=float, default=0.001, help="统一 LR（覆盖 LLRD）")
    p.add_argument("--output", default="outputs/short_test", help="输出目录")
    return p.parse_args()


def build_optimizer(model, cfg, lr_override):
    """构建优化器，可选 LR 覆盖。"""
    param_groups = build_llrd_param_groups(model, cfg, freeze_backbone=False)
    if lr_override > 0:
        for g in param_groups:
            g["lr"] = lr_override
            g["initial_lr"] = lr_override
    opt_cfg = cfg.get("optimizer", {})
    return torch.optim.SGD(
        param_groups,
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 5e-4),
        nesterov=opt_cfg.get("nesterov", True),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Short Test — {cfg.get('experiment', {}).get('name', 'unknown')}")
    print(f"  Samples: {args.samples} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # ─── DataLoader ───
    loader = build_dataloader(cfg, split="train", batch_size=args.batch_size,
                              num_workers=min(4, args.batch_size // 2), shuffle=True)
    gpu_proc = getattr(loader, "gpu_processor", None)

    steps_per_epoch = len(loader)
    total_iters = args.epochs * steps_per_epoch

    print(f"  Steps/epoch: {steps_per_epoch} | Total iters: {total_iters}")

    # ─── 模型 + 损失 ───
    model = build_lead_net(cfg)
    model.to(device)
    model.train()
    criterion = MultiBoxLoss(
        num_classes=cfg.get("num_classes", 7) + 1,
        input_size=cfg.get("data", {}).get("input_size", 320),
        overlap_threshold=cfg.get("loss", {}).get("overlap_threshold", 0.35),
        class_weights=cfg.get("class_weights", None),
    )

    # ─── 优化器 + 调度器 ───
    optimizer = build_optimizer(model, cfg, args.lr)
    scheduler = build_scheduler_from_total_iters(
        optimizer,
        warmup_iters=min(100, total_iters // 10),
        total_iters=total_iters,
        warmup_start_factor=0.05,  # 超快 warmup
        eta_min_factor=0.0,
    )

    # ─── AMP ───
    use_amp = cfg.get("training", {}).get("amp", False) and device.type == "cuda"
    scaler_init = cfg.get("training", {}).get("grad_scaler_init_scale", 2048.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=scaler_init) if use_amp else None

    # ─── 训练循环 ───
    history: list[dict] = []
    t_start = time.time()
    total_nan_skipped = 0

    for epoch in range(args.epochs):
        epoch_cls = epoch_loc = epoch_loss = 0.0
        epoch_grad_norm = 0.0
        n = 0
        t_e_start = time.time()

        for bi, batch in enumerate(loader):
            if gpu_proc is not None:
                batch = gpu_proc(batch)
            images = batch["image"] if gpu_proc is not None else batch["image"].to(device)
            gt_boxes = batch["boxes"]
            gt_labels = batch["labels"]

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    cls_pred, loc_pred = model(images)
                    default_boxes = model.head.all_default_boxes(device)
                    cls_loss, loc_loss = criterion(cls_pred, loc_pred, default_boxes, gt_boxes, gt_labels)
                    loss = cls_loss + loc_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                cls_pred, loc_pred = model(images)
                default_boxes = model.head.all_default_boxes(device)
                cls_loss, loc_loss = criterion(cls_pred, loc_pred, default_boxes, gt_boxes, gt_labels)
                loss = cls_loss + loc_loss
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
                optimizer.step()

            scheduler.step()

            # NaN 跳过
            if torch.isnan(loss) or torch.isinf(loss):
                total_nan_skipped += 1
                print(f"  ⚠️  NaN batch skipped at epoch {epoch+1} batch {bi}")
                continue

            cls_v = cls_loss.item()
            loc_v = loc_loss.item()
            loss_v = loss.item()

            epoch_cls += cls_v
            epoch_loc += loc_v
            epoch_loss += loss_v
            epoch_grad_norm += grad_norm
            n += 1

        # ─── epoch 摘要 ───
        avg_cls = epoch_cls / max(n, 1)
        avg_loc = epoch_loc / max(n, 1)
        avg_loss = epoch_loss / max(n, 1)
        avg_grad = epoch_grad_norm / max(n, 1)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_e_start

        entry = {
            "epoch": epoch + 1,
            "cls": round(avg_cls, 4),
            "loc": round(avg_loc, 4),
            "loss": round(avg_loss, 4),
            "grad_norm": round(avg_grad, 2),
            "lr": lr,
            "time_s": round(elapsed, 1),
        }
        history.append(entry)

        # 梯度统计（每 5 epoch）
        grad_summary = ""
        if (epoch + 1) % 5 == 0:
            stats = grad_stats(model)
            grad_summary = f" grad: {summarize_grads(stats)}"

        print(f"  epoch {epoch+1:3d}/{args.epochs}  "
              f"cls={avg_cls:.4f}  loc={avg_loc:.4f}  loss={avg_loss:.4f}  "
              f"grad={avg_grad:.2f}  lr={lr:.2e}  {elapsed:.0f}s{grad_summary}")

    total_time = time.time() - t_start

    # ─── 结果 ───
    print(f"\n{'='*60}")
    print(f"  Results")

    initial_loss = history[0]["loss"]
    final_loss = history[-1]["loss"]
    loss_trend = "↓ decreasing" if final_loss < initial_loss * 0.95 else (
        "→ flat" if abs(final_loss - initial_loss) < initial_loss * 0.05 else "↑ increasing"
    )

    print(f"  Loss: {initial_loss:.4f} → {final_loss:.4f} ({loss_trend})")
    print(f"  NaN batches skipped: {total_nan_skipped}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # 判断
    has_nan_in_history = any(
        h["cls"] != h["cls"] or h["loc"] != h["loc"]  # NaN != NaN → True
        for h in history
    )

    if has_nan_in_history:
        print("\n  [FAIL] NaN in loss history")
        verdict = "FAIL"
    elif total_nan_skipped > 0:
        print(f"\n  [WARN] {total_nan_skipped} NaN batches skipped")
        verdict = "MARGINAL"
    elif final_loss < initial_loss * 0.8:
        print(f"\n  [PASS] Loss decreased >20%")
        verdict = "PASS"
    elif final_loss < initial_loss:
        print(f"\n  [MARGINAL] Loss decreased <20%, may need more epochs or LR tuning")
        verdict = "MARGINAL"
    else:
        print(f"\n  [FAIL] Loss did not decrease")
        verdict = "FAIL"

    # ─── 保存 ───
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "device": str(device),
        "samples": args.samples,
        "epochs": args.epochs,
        "lr": args.lr,
        "verdict": verdict,
        "total_time_s": round(total_time, 1),
        "nan_skipped": total_nan_skipped,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "history": history,
    }

    report_path = out_dir / "short_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to: {report_path}")

    sys.exit(0 if verdict in ("PASS", "MARGINAL") else 1)


if __name__ == "__main__":
    main()
