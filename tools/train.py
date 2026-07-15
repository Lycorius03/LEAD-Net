"""训练入口（云端 RTX 5090 专用，两阶段 LLRD + EMA + Auto Batch）。

用法:
    python tools/train.py --config configs/train_baseline.yaml
    python tools/train.py --config configs/train_lca.yaml
"""

from __future__ import annotations

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lead_net.models import build_lead_net
from lead_net.models.loss import MultiBoxLoss
from lead_net.data import build_dataloader
from lead_net.engine import Trainer, MetricsCollector
from lead_net.utils import load_config, resolve_paths_in, ExperimentManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net 云端训练")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--max-samples", type=int, default=None, help="限制样本数(冒烟测试)")
    p.add_argument("--stage1-only", action="store_true", help="仅执行阶段一")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = resolve_paths_in(cfg)

    dev = cfg.get("device", {}).get("type", "cuda")
    device = torch.device(dev if (dev == "cuda" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"[info] {torch.cuda.get_device_name(device)} ({gb:.1f}GB)")

    # 数据
    nw = 0 if sys.platform == "win32" else cfg.get("train", {}).get("num_workers", 4)
    train_loader = build_dataloader(cfg, split="train", num_workers=nw)
    val_loader = build_dataloader(cfg, split="val", num_workers=nw)
    if args.max_samples:
        ds = train_loader.dataset
        ds.ids = ds.ids[:min(args.max_samples, len(ds.ids))]
        print(f"[info] limited to {len(ds.ids)} train samples")
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

    summary = trainer.fit()
    print(f"[info] done: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
