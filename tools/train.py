"""训练入口（云端 RTX 5090 专用，两阶段 LLRD + EMA + Warmup-Cosine）。

用法:
    python tools/train.py --config configs/train_lca.yaml
    python tools/train.py --config configs/train_lca.yaml --smoke       # 冒烟测试(少量图片)
    python tools/train.py --config configs/train_lca.yaml --resume      # 从 checkpoint 恢复
"""

from __future__ import annotations

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lead_net.models import build_lead_net
from lead_net.models.loss import MultiBoxLoss
from lead_net.data import build_dataloader
from lead_net.engine import Trainer, MetricsCollector, CheckpointManager
from lead_net.utils import load_config, resolve_paths_in, ExperimentManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net 云端训练")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--max-samples", type=int, default=None, help="限制样本数(冒烟测试)")
    p.add_argument("--smoke", action="store_true", help="冒烟测试模式(2张图,1 epoch)")
    p.add_argument("--stage1-only", action="store_true", help="仅执行阶段一")
    p.add_argument("--resume", action="store_true", help="从 latest checkpoint 恢复训练")
    p.add_argument("--device", type=str, default=None, help="设备覆盖(cuda/cpu)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = resolve_paths_in(cfg)

    # 设备
    dev = args.device or cfg.get("device", {}).get("type", "cuda")
    device = torch.device(dev if (dev == "cuda" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"[info] {torch.cuda.get_device_name(device)} ({gb:.1f}GB)")

    # ── 冒烟测试模式 ──
    if args.smoke:
        args.max_samples = 2
        cfg["stage2_joint_training"]["epochs"] = 1
        cfg["stage1_freeze_backbone"]["enabled"] = False  # 冒烟直接联合训练
        cfg["eval"]["eval_interval"] = 1
        print("[smoke] 冒烟测试模式: 2 张图, 1 epoch, 跳过 Stage1")

    # 数据
    import sys as _sys
    nw = 0 if _sys.platform == "win32" else (cfg.get("training") or cfg.get("train", {})).get("num_workers", 4)
    train_loader = build_dataloader(cfg, split="train", num_workers=nw)
    val_loader = build_dataloader(cfg, split="val", num_workers=nw)

    if args.max_samples:
        ds = train_loader.dataset
        # 兼容 COCO dataset (ids) 和 TXT dataset (_image_paths)
        if hasattr(ds, "ids"):
            ds.ids = ds.ids[:min(args.max_samples, len(ds.ids))]
        elif hasattr(ds, "_image_paths"):
            ds._image_paths = ds._image_paths[:min(args.max_samples, len(ds._image_paths))]
        elif hasattr(ds, "_ids"):
            ds._ids = ds._ids[:min(args.max_samples, len(ds._ids))]
        n_samples = len(ds)
        print(f"[info] limited to {n_samples} train samples")

        # 确保 batch_size 不超过样本数（否则 drop_last 会丢弃所有数据）
        bs = cfg.get("training", {}).get("batch_size", 16)
        if bs > n_samples:
            cfg["training"]["batch_size"] = max(1, n_samples)
            # 重建 loader → 重新限制样本数
            train_loader = build_dataloader(cfg, split="train", num_workers=nw)
            ds2 = train_loader.dataset
            if hasattr(ds2, "ids"):
                ds2.ids = ds2.ids[:min(args.max_samples, len(ds2.ids))]
            elif hasattr(ds2, "_image_paths"):
                ds2._image_paths = ds2._image_paths[:min(args.max_samples, len(ds2._image_paths))]
            elif hasattr(ds2, "_ids"):
                ds2._ids = ds2._ids[:min(args.max_samples, len(ds2._ids))]
            print(f"[info] batch_size adjusted to {cfg['training']['batch_size']} (was {bs})")

    print(f"[info] train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    # 模型
    model = build_lead_net(cfg)
    print(f"[info] params: {sum(p.numel() for p in model.parameters()):,}")

    # 实验管理
    session = ExperimentManager.next_session("outputs/experiments")
    variant = "lca" if cfg.get("model", {}).get("lca", {}).get("enabled", False) else "baseline"
    mgr = ExperimentManager("outputs/experiments", session, variant)
    mgr.setup(cfg)
    print(f"[info] {mgr.run_dir}")

    # 采集 + 损失
    tag = cfg.get("experiment", {}).get("tag", variant)
    collector = MetricsCollector(output_dir=str(mgr.run_dir), experiment_tag=tag)
    num_cls = cfg.get("num_classes", 80) + 1
    criterion = MultiBoxLoss(num_classes=num_cls, input_size=cfg.get("data", {}).get("input_size", 320))

    # 训练
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, cfg=cfg, device=device,
        output_dir=mgr.checkpoint_dir, collector=collector,
    )

    if args.stage1_only:
        cfg["stage2_joint_training"]["epochs"] = 0

    # ── 恢复或新训练 ──
    if args.resume:
        summary = trainer.fit_resume()
    else:
        summary = trainer.fit()

    print(f"[info] done: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
