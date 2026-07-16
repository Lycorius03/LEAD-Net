#!/usr/bin/env python
"""Overfit 测试 —— 单 batch 过拟合验证。

目的：
    验证模型 + 损失 + 优化器 + AMP 在极小数据上能正常收敛。
    如果 50 iter 后 loss 不降，说明存在严重 bug（非超参问题）。

用法:
    python tools/overfit_test.py --config configs/train_lca.yaml
    python tools/overfit_test.py --config configs/train_lca.yaml --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lead_net.utils.config import load_config
from lead_net.models.lead_net import build_lead_net
from lead_net.models.loss import MultiBoxLoss
from lead_net.data import build_dataloader, collate_fn
from lead_net.engine.llrd import build_llrd_param_groups


def parse_args():
    p = argparse.ArgumentParser(description="Overfit test — single batch convergence")
    p.add_argument("--config", required=True, help="训练配置文件")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--iters", type=int, default=50, help="迭代次数")
    p.add_argument("--batch-size", type=int, default=4, help="过拟合用 batch 大小")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_cfg = cfg.get("training", {})

    # Override batch size
    train_cfg["batch_size"] = args.batch_size

    print(f"=== Overfit Test ===")
    print(f"  Config: {args.config}")
    print(f"  Device: {device}")
    print(f"  Iters:  {args.iters}")
    print(f"  Batch:  {args.batch_size}")

    # ─── DataLoader ───
    loader = build_dataloader(cfg, split="train", batch_size=args.batch_size,
                              num_workers=0, shuffle=False)
    gpu_proc = getattr(loader, "gpu_processor", None)

    # ─── 模型 ───
    model = build_lead_net(cfg)
    model.to(device)
    model.train()

    # ─── 损失 ───
    criterion = MultiBoxLoss(cfg)

    # ─── 优化器 ───
    param_groups = build_llrd_param_groups(model, cfg, freeze_backbone=False)
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.SGD(
        param_groups,
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 5e-4),
        nesterov=opt_cfg.get("nesterov", True),
    )

    # ─── AMP ───
    use_amp = train_cfg.get("amp", False) and device.type == "cuda"
    scaler_init = train_cfg.get("grad_scaler_init_scale", 2048.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=scaler_init) if use_amp else None

    # ─── 取固定 batch ───
    batch = next(iter(loader))
    if gpu_proc is not None:
        batch = gpu_proc(batch)
    images = batch["image"] if gpu_proc is not None else batch["image"].to(device)
    gt_boxes = batch["boxes"]
    gt_labels = batch["labels"]

    n_boxes = [len(b) for b in gt_boxes]
    print(f"  Images: {images.shape}, Boxes: {n_boxes}, Total boxes: {sum(n_boxes)}")

    # ─── Overfit Loop ───
    t_start = time.time()
    losses = []
    grad_norms = []
    nan_detected = False

    for i in range(args.iters):
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                cls_pred, loc_pred = model(images)
                default_boxes = model.head.all_default_boxes(device)
                cls_loss, loc_loss = criterion(cls_pred, loc_pred, default_boxes, gt_boxes, gt_labels)
                loss = cls_loss + loc_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
            scaler.step(optimizer)
            scaler.update()
        else:
            cls_pred, loc_pred = model(images)
            default_boxes = model.head.all_default_boxes(device)
            cls_loss, loc_loss = criterion(cls_pred, loc_pred, default_boxes, gt_boxes, gt_labels)
            loss = cls_loss + loc_loss
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
            optimizer.step()

        cls_v = cls_loss.item()
        loc_v = loc_loss.item()
        loss_v = loss.item()

        losses.append({"iter": i, "cls": cls_v, "loc": loc_v, "total": loss_v})
        grad_norms.append(gnorm)

        # NaN 检查
        if math.isnan(loss_v) or math.isinf(loss_v):
            print(f"  ❌ NaN/Inf at iter {i}")
            nan_detected = True
            break

        if i == 0 or (i + 1) % 10 == 0 or i == args.iters - 1:
            print(f"  iter {i+1:3d}/{args.iters}  cls={cls_v:.4f}  loc={loc_v:.4f}  "
                  f"loss={loss_v:.4f}  grad={gnorm:.2f}  lr={optimizer.param_groups[0]['lr']:.2e}")

    elapsed = time.time() - t_start

    # ─── 结果判断 ───
    print(f"\n--- Results ---")
    initial_loss = losses[0]["total"]
    final_loss = losses[-1]["total"]
    loss_ratio = final_loss / max(initial_loss, 1e-8)
    avg_grad = sum(grad_norms) / max(len(grad_norms), 1)

    print(f"  Initial loss:   {initial_loss:.4f}")
    print(f"  Final loss:     {final_loss:.4f}")
    print(f"  Ratio:          {loss_ratio:.3f}")
    print(f"  Avg grad norm:  {avg_grad:.2f}")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"  NaN detected:   {nan_detected}")

    if nan_detected:
        print("\n  [FAIL] NaN detected during overfit test")
        sys.exit(1)
    elif loss_ratio > 0.95:
        print(f"\n  [WARN] Loss barely decreased (ratio={loss_ratio:.3f})")
        sys.exit(1)
    elif loss_ratio < 0.5:
        print(f"\n  [PASS] Loss decreased significantly (ratio={loss_ratio:.3f})")
        sys.exit(0)
    else:
        print(f"\n  [MARGINAL] Some decrease (ratio={loss_ratio:.3f}), "
              f"consider more iterations or check LR")
        sys.exit(0)


if __name__ == "__main__":
    import math
    main()
