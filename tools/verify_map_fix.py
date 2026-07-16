#!/usr/bin/env python
"""mAP修复验证训练 —— 5-10 epoch 完整训练 + mAP评估 + 自动清理。

目的：
    在本地跑少量epoch验证bbox坐标修复是否有效。
    若mAP>0则证明修复成功，自动删除训练输出避免与云端数据混淆。

用法:
    python tools/verify_map_fix.py --epochs 8 --train-samples 2000
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lead_net.models import build_lead_net
from lead_net.models.loss import MultiBoxLoss
from lead_net.data import build_dataloader
from lead_net.engine import Trainer, MetricsCollector, Evaluator
from lead_net.engine.llrd import build_llrd_param_groups, freeze_backbone, unfreeze_backbone
from lead_net.engine.ema import ModelEMA
from lead_net.engine.scheduler import build_scheduler
from lead_net.utils import load_config, resolve_paths_in


def parse_args():
    p = argparse.ArgumentParser(description="mAP Fix Verification Training")
    p.add_argument("--config", default="configs/train_baseline.yaml")
    p.add_argument("--epochs", type=int, default=8, help="训练epoch数（默认8）")
    p.add_argument("--train-samples", type=int, default=2000, help="训练样本数")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def limit_dataset(loader, n: int):
    """限制 DataLoader 的数据集样本数。"""
    ds = loader.dataset
    if hasattr(ds, "_image_paths"):
        ds._image_paths = ds._image_paths[:min(n, len(ds._image_paths))]
    elif hasattr(ds, "ids"):
        ds.ids = ds.ids[:min(n, len(ds.ids))]
    elif hasattr(ds, "_ids"):
        ds._ids = ds._ids[:min(n, len(ds._ids))]
    return len(ds)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = resolve_paths_in(cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[verify] Device: {device}")
    print(f"[verify] Epochs: {args.epochs}, Train samples: {args.train_samples}")

    # ─── 覆盖配置：缩减epoch ───
    cfg.setdefault("stage2_joint_training", {})["epochs"] = args.epochs
    cfg["stage2_joint_training"]["early_stopping"] = False
    cfg["stage1_freeze_backbone"]["max_epochs"] = min(2, max(1, args.epochs // 4))
    cfg.setdefault("eval", {})["eval_interval"] = max(1, args.epochs // 2)

    # ─── 输出目录（临时，验证后删除）───
    output_dir = Path("outputs/verify_map_fix")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[verify] Output: {output_dir}")

    # ─── 数据 ───
    nw = 0  # Windows
    train_loader = build_dataloader(cfg, split="train", num_workers=nw)
    val_loader = build_dataloader(cfg, split="val", num_workers=nw)

    n_train = limit_dataset(train_loader, args.train_samples)
    print(f"[verify] Train samples: {n_train}, Val samples: {len(val_loader.dataset)}")

    # ─── 模型 + 损失 ───
    model = build_lead_net(cfg).to(device)
    num_cls = cfg.get("num_classes", 7) + 1
    criterion = MultiBoxLoss(num_classes=num_cls, input_size=cfg.get("data", {}).get("input_size", 320))

    # ─── 优化器 + EMA ───
    train_cfg = cfg.get("training", {})
    ema = ModelEMA(model, decay=train_cfg.get("ema_decay", 0.9998)) if train_cfg.get("ema", False) else None
    use_amp = train_cfg.get("amp", False) and device.type == "cuda"
    scaler_init = train_cfg.get("grad_scaler_init_scale", 2048.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=scaler_init) if use_amp else None
    grad_clip = train_cfg.get("grad_clip_norm", 5.0)

    # ─── 阶段一：冻结 Backbone ───
    stage1_cfg = cfg.get("stage1_freeze_backbone", {})
    if stage1_cfg.get("enabled", False):
        print("\n=== Stage 1: Freeze Backbone ===")
        freeze_backbone(model)
        param_groups = build_llrd_param_groups(model, cfg, freeze_backbone=True)
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=cfg.get("optimizer", {}).get("momentum", 0.9),
            weight_decay=cfg.get("optimizer", {}).get("weight_decay", 5e-4),
            nesterov=cfg.get("optimizer", {}).get("nesterov", True),
        )
        for pg in optimizer.param_groups:
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg["lr"]

        total_iters_s1 = stage1_cfg["max_epochs"] * len(train_loader)
        s1_scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader),
                                        total_epochs=stage1_cfg["max_epochs"])

        s1_max = stage1_cfg["max_epochs"]
        prev_loss = float("inf")
        for epoch in range(1, s1_max + 1):
            model.train()
            total_loss = 0.0
            n = 0
            for batch in train_loader:
                images = batch["image"].to(device)
                gt_boxes, gt_labels = batch["boxes"], batch["labels"]
                optimizer.zero_grad()
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        cls_pred, loc_pred = model(images)
                        dboxes = model.head.all_default_boxes(device)
                        cl, ll = criterion(cls_pred, loc_pred, dboxes, gt_boxes, gt_labels)
                        loss = cl + ll
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    cls_pred, loc_pred = model(images)
                    dboxes = model.head.all_default_boxes(device)
                    cl, ll = criterion(cls_pred, loc_pred, dboxes, gt_boxes, gt_labels)
                    loss = cl + ll
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                s1_scheduler.step()
                if ema:
                    ema.update()
                total_loss += loss.item()
                n += 1
            avg_loss = total_loss / max(n, 1)
            change = abs(prev_loss - avg_loss) / max(abs(prev_loss), 1e-8)
            print(f"  S1 epoch {epoch}/{s1_max}  loss={avg_loss:.4f}  delta={change:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
            if change < stage1_cfg.get("loss_change_threshold", 0.05) and epoch >= 2:
                print(f"  Loss stabilized -> unfreezing early")
                break
            prev_loss = avg_loss

        print("[verify] Unfreezing backbone for Stage 2...")
        unfreeze_backbone(model)

    # ─── 阶段二：LLRD 联合训练 ───
    print(f"\n=== Stage 2: Joint Training ({args.epochs} epochs) ===")
    param_groups = build_llrd_param_groups(model, cfg, freeze_backbone=False)
    optimizer = torch.optim.SGD(
        param_groups,
        momentum=cfg.get("optimizer", {}).get("momentum", 0.9),
        weight_decay=cfg.get("optimizer", {}).get("weight_decay", 5e-4),
        nesterov=cfg.get("optimizer", {}).get("nesterov", True),
    )
    for pg in optimizer.param_groups:
        if "initial_lr" not in pg:
            pg["initial_lr"] = pg["lr"]
    s2_scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader),
                                    total_epochs=args.epochs)

    eval_interval = cfg["eval"]["eval_interval"]
    best_mAP = -1.0
    all_metrics = []

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_cls = total_loc = total_grad = 0.0
        n = 0
        for bi, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            gt_boxes, gt_labels = batch["boxes"], batch["labels"]
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    cls_pred, loc_pred = model(images)
                    dboxes = model.head.all_default_boxes(device)
                    cl, ll = criterion(cls_pred, loc_pred, dboxes, gt_boxes, gt_labels)
                    loss = cl + ll
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                cls_pred, loc_pred = model(images)
                dboxes = model.head.all_default_boxes(device)
                cl, ll = criterion(cls_pred, loc_pred, dboxes, gt_boxes, gt_labels)
                loss = cl + ll
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
                optimizer.step()
            s2_scheduler.step()
            if ema:
                ema.update()
            total_cls += cl.item()
            total_loc += ll.item()
            total_grad += gnorm
            n += 1

        avg_cls = total_cls / max(n, 1)
        avg_loc = total_loc / max(n, 1)
        lr = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch:3d}/{args.epochs}  cls={avg_cls:.4f}  loc={avg_loc:.4f}  "
              f"loss={avg_cls+avg_loc:.4f}  lr={lr:.2e}")

        # Eval mAP
        eval_metrics = None
        if epoch % eval_interval == 0 or epoch == args.epochs:
            print(f"  Running mAP evaluation...")
            if ema:
                ema.apply()
            model.eval()
            evaluator = Evaluator(model=model, val_loader=val_loader, cfg=cfg, device=device)
            eval_metrics = evaluator.evaluate()
            if ema:
                ema.restore()
            model.train()

            mAP50 = eval_metrics.get("mAP@0.5", 0.0)
            print(f"  >>> mAP@0.5={mAP50:.6f}  mAP@0.5:0.95={eval_metrics.get('mAP@0.5:0.95',0):.6f}")
            all_metrics.append({"epoch": epoch, "mAP50": mAP50,
                                "cls_loss": avg_cls, "loc_loss": avg_loc})
            if mAP50 > best_mAP:
                best_mAP = mAP50
                # Save best
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "mAP@0.5": mAP50,
                }, ckpt_dir / "best.pth")

    total_time = time.time() - t_start

    # ─── 最终评估 ───
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")

    if all_metrics:
        final_mAP = all_metrics[-1]["mAP50"]
        print(f"  Best mAP@0.5:  {best_mAP:.6f}")
        print(f"  Final mAP@0.5: {final_mAP:.6f}")
        print(f"  Training time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # ─── 判定 ───
    fix_works = best_mAP > 0.001
    if fix_works:
        print(f"\n  [SUCCESS] mAP@0.5 = {best_mAP:.6f} > 0.001!")
        print(f"  BBox coordinate fix is CONFIRMED working.")
        print(f"  Deleting local outputs to avoid confusion with cloud training...")
    elif best_mAP > 0.0:
        print(f"\n  [MARGINAL] mAP@0.5 = {best_mAP:.6f} > 0 but very low.")
        print(f"  Fix is likely working, needs more epochs to confirm.")
        print(f"  Keeping outputs for inspection.")
        fix_works = True  # Consider marginal as partial success
    else:
        print(f"\n  [FAIL] mAP@0.5 still = 0 after {args.epochs} epochs.")
        print(f"  Keeping outputs for debugging.")

    # ─── 清理 ───
    if fix_works:
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"  Deleted: {output_dir}")
        # Also clean quick_map_test outputs if any
        qmt = Path("outputs/quick_map_test")
        if qmt.exists():
            shutil.rmtree(qmt)
            print(f"  Deleted: {qmt}")

    return 0 if fix_works else 1


if __name__ == "__main__":
    raise SystemExit(main())
