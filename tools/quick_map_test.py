#!/usr/bin/env python
"""快速 mAP 验证 —— 500 张图, 20 epoch, 每 5 epoch 测 mAP。

目的:
    在短时间(~30min)内看到 mAP 趋势, 验证修改是否有改善。
    不跑完整训练, 仅用于快速诊断。

用法:
    python tools/quick_map_test.py --config configs/train_lca.yaml
    python tools/quick_map_test.py --config configs/train_lca.yaml --baseline
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

from lead_net.utils import load_config, resolve_paths_in
from lead_net.models import build_lead_net
from lead_net.models.loss import MultiBoxLoss
from lead_net.data import build_dataloader, collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="Quick mAP validation smoke test")
    p.add_argument("--config", required=True, help="配置文件")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-samples", type=int, default=500, help="训练图片数")
    p.add_argument("--val-samples", type=int, default=200, help="验证图片数")
    p.add_argument("--epochs", type=int, default=20, help="训练 epoch")
    p.add_argument("--eval-every", type=int, default=5, help="每 N epoch 测 mAP")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--baseline", action="store_true", help="对比: 用旧锚框+旧IoU阈值")
    return p.parse_args()


def build_quick_loader(cfg, split, n_samples, batch_size, device_type):
    """构建小样本 dataloader。"""
    loader = build_dataloader(cfg, split=split, batch_size=batch_size,
                              num_workers=0, shuffle=(split == "train"))
    ds = loader.dataset
    # 随机采样
    import random
    indices = random.Random(42).sample(range(len(ds)), min(n_samples, len(ds)))

    subset = Subset(ds, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=(split == "train"),
                      collate_fn=collate_fn, drop_last=(split == "train"))


def evaluate_map(model, val_loader, cfg, device):
    """在验证子集上计算 mAP。"""
    from lead_net.engine.evaluator import Evaluator
    model.eval()
    evaluator = Evaluator(model, val_loader, cfg, device)
    result = evaluator.evaluate()
    model.train()
    return result


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = resolve_paths_in(cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] {torch.cuda.get_device_name(device)}")

    # ── Baseline 模式: 回退旧设置 ──
    if args.baseline:
        print("[info] BASELINE MODE: 旧锚框 (2,475) + IoU=0.5 + 无类别权重")
        # 不修改 cfg, 直接在 loss 创建时覆盖
        cfg_override = {"overlap_threshold": 0.5, "class_weights": None}
        # 注意: 锚框无法在运行时覆盖, baseline 用新锚框+旧IoU作为"半对照"
        print("[warn] 锚框已更新为 4,725, baseline 仅回退 IoU+weights")
    else:
        cfg_override = {
            "overlap_threshold": cfg.get("loss", {}).get("overlap_threshold", 0.35),
            "class_weights": cfg.get("class_weights", None),
        }

    # ── 数据 ──
    train_loader = build_quick_loader(cfg, "train", args.train_samples,
                                       args.batch_size, device.type)
    val_loader_full = build_quick_loader(cfg, "val", args.val_samples,
                                           args.batch_size, device.type)
    steps_per_epoch = len(train_loader)

    num_classes = cfg.get("num_classes", 7)
    input_size = cfg.get("data", {}).get("input_size", 320)

    # 诊断: 训练子集锚框匹配统计
    print(f"\n--- 训练子集锚框匹配诊断 ---")
    print(f"  Samples: {args.train_samples}, Batch: {args.batch_size}")
    total_pos_all = 0
    total_gt_all = 0
    per_cls_pos = {}
    zero_count = 0
    with torch.no_grad():
        model_tmp = build_lead_net(cfg)
        dboxes = model_tmp.head.all_default_boxes(device)
        for batch in train_loader:
            stats = MultiBoxLoss.diagnose_matching(
                dboxes, batch["boxes"], batch["labels"],
                input_size=input_size,
                overlap_threshold=cfg_override["overlap_threshold"],
            )
            total_pos_all += stats["avg_pos_per_img"] * max(len(batch["boxes"]), 1)
            total_gt_all += stats["total_gt"]
            for k, v in stats["per_class_pos"].items():
                per_cls_pos[k] = per_cls_pos.get(k, 0) + v
            zero_count += int(stats["pct_imgs_zero_pos"] / 100 * max(len(batch["boxes"]), 1))
    n_imgs = args.train_samples
    print(f"  Avg pos/img: {total_pos_all/max(n_imgs,1):.1f}")
    print(f"  Zero-pos imgs: {zero_count}/{n_imgs} ({zero_count/max(n_imgs,1)*100:.1f}%)")
    print(f"  Per-class pos: {dict(sorted(per_cls_pos.items()))}")

    # ── 模型 ──
    model = build_lead_net(cfg).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    n_anchors = model.head.all_default_boxes(device).shape[0]
    print(f"\n  Model: {n_params:,} params, {n_anchors} anchors")

    # ── 损失 ──
    criterion = MultiBoxLoss(
        num_classes=num_classes + 1,
        input_size=input_size,
        overlap_threshold=cfg_override["overlap_threshold"],
        class_weights=cfg_override["class_weights"],
    )

    # ── 优化器 ──
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.003,
        momentum=0.9,
        weight_decay=0.0005,
    )
    # 简易 Cosine 调度 (warmup 后 cosine)
    warmup_iters = 50
    total_iters = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step / max(1, warmup_iters)) * 0.01 + 0.99
        if step < warmup_iters
        else 0.5 * (1 + math.cos(math.pi * (step - warmup_iters) / max(1, total_iters - warmup_iters))),
    )

    # ── 训练 ──
    gpu_proc = getattr(train_loader, "gpu_processor", None)
    results = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_cls, total_loc, total_grad, n = 0.0, 0.0, 0.0, 0

        for bi, batch in enumerate(train_loader):
            if gpu_proc is not None:
                batch = gpu_proc(batch)
            images = batch["image"] if gpu_proc is not None else batch["image"].to(device)

            optimizer.zero_grad()
            cls_pred, loc_pred = model(images)
            dboxes = model.head.all_default_boxes(device)
            cl, ll = criterion(cls_pred, loc_pred, dboxes,
                               batch["boxes"], batch["labels"])
            loss = cl + ll
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
            optimizer.step()
            scheduler.step()

            total_cls += cl.item()
            total_loc += ll.item()
            total_grad += gnorm
            n += 1

        avg_cls = total_cls / n
        avg_loc = total_loc / n
        avg_loss = avg_cls + avg_loc
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start

        # 进度
        do_eval = (epoch % args.eval_every == 0 or epoch == args.epochs)
        map_str = ""
        map50 = None
        if do_eval:
            print(f"  evaluating mAP on {args.val_samples} val images...", end="", flush=True)
            eval_t0 = time.time()
            map_result = evaluate_map(model, val_loader_full, cfg, device)
            map50 = map_result.get("mAP@0.5", 0)
            map5095 = map_result.get("mAP@0.5:0.95", 0)
            eval_time = time.time() - eval_t0
            map_str = f"  mAP@0.5={map50:.4f}  mAP@0.5:0.95={map5095:.4f}  ({eval_time:.0f}s)"
            results.append({
                "epoch": epoch, "cls_loss": avg_cls, "loc_loss": avg_loc,
                "mAP@0.5": map50, "mAP@0.5:0.95": map5095,
            })

        print(f"  epoch {epoch:2d}/{args.epochs}  "
              f"cls={avg_cls:.4f}  loc={avg_loc:.4f}  loss={avg_loss:.4f}  "
              f"grad={total_grad/n:.2f}  lr={lr:.2e}  {elapsed:.0f}s"
              f"{'  ' + map_str if map_str else ''}", flush=True)

    # ── 最终报告 ──
    print(f"\n{'='*60}")
    print(f"  mAP Trend")
    print(f"{'='*60}")
    mode = "BASELINE (IoU=0.5, no weights)" if args.baseline else "NEW (IoU=0.35, weights, new anchors)"
    print(f"  Mode: {mode}")
    print(f"  {'Epoch':<8} {'cls':<10} {'loc':<10} {'mAP@0.5':<12} {'mAP@0.5:0.95':<14}")
    print(f"  {'-'*54}")
    for r in results:
        print(f"  {r['epoch']:<8} {r['cls_loss']:<10.4f} {r['loc_loss']:<10.4f} "
              f"{r['mAP@0.5']:<12.4f} {r['mAP@0.5:0.95']:<14.4f}")

    final_map = results[-1]["mAP@0.5"] if results else 0.0
    # 判定
    if final_map > 0.01:
        verdict = "GOOD — mAP > 1%, 模型正在学习"
    elif final_map > 0.001:
        verdict = "WEAK — mAP < 1% but > 0.1%, 可能需要更多 epoch"
    else:
        verdict = "POOR — mAP ≈ 0, 需要进一步诊断"

    print(f"\n  Verdict: {verdict}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
