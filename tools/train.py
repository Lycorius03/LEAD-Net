"""训练入口脚本。

用法：
    python tools/train.py --config configs/baseline_ssd.yaml
    python tools/train.py --config configs/baseline_ssd.yaml --epochs 5 --max-samples 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lead_net.models import build_lead_net
from lead_net.models.loss import MultiBoxLoss
from lead_net.data import build_dataloader
from lead_net.engine import Trainer, MetricsCollector
from lead_net.utils import load_config, resolve_paths_in, ensure_dir, get_nested


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net 训练入口（训练 + 验证 + CSV 指标采集）")
    p.add_argument("--config", required=True, type=str,
                   help="配置文件路径（如 configs/baseline_ssd.yaml）")
    p.add_argument("--epochs", type=int, default=None,
                   help="覆盖配置中的 epochs（用于快速测试）")
    p.add_argument("--batch-size", type=int, default=None,
                   help="覆盖 batch_size")
    p.add_argument("--max-samples", type=int, default=None,
                   help="限制数据集样本数（用于快速测试）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = resolve_paths_in(cfg)

    if args.epochs:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size

    dev_type = get_nested(cfg, "device.type", "cuda")
    device = torch.device(dev_type if (dev_type == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[info] device: {device}")

    # ---- 数据 ----
    num_workers = 0 if sys.platform == "win32" else get_nested(cfg, "train.num_workers", 4)
    train_loader = build_dataloader(cfg, split="train", num_workers=num_workers)
    val_loader = build_dataloader(cfg, split="val", num_workers=num_workers)

    if args.max_samples:
        _limit_dataset(train_loader.dataset, args.max_samples)
        _limit_dataset_cfg(cfg, args.max_samples)

    print(f"[info] train samples: {len(train_loader.dataset)}")
    print(f"[info] val samples:   {len(val_loader.dataset)}")

    # ---- 模型 ----
    model = build_lead_net(cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model params: {n_params:,}")

    # ---- 损失函数 ----
    num_classes = cfg.get("num_classes", 80) + 1
    input_size = cfg.get("data", {}).get("input_size", 320)
    criterion = MultiBoxLoss(num_classes=num_classes, input_size=input_size)

    # ---- 优化器 ----
    train_cfg = cfg.get("train", {})
    opt_cfg = train_cfg.get("optimizer", {})
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt_cfg.get("lr", 0.01),
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 5e-4),
    )

    # ---- 学习率调度器 ----
    scheduler = None
    sch_cfg = train_cfg.get("lr_scheduler", {})
    if sch_cfg.get("name") == "cosine":
        epochs = args.epochs or train_cfg.get("epochs", 50)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---- 输出目录 ----
    ckpt_dir = get_nested(cfg, "paths.checkpoint_dir", "outputs/checkpoints")
    ensure_dir(ckpt_dir)
    experiment_csv_dir = get_nested(cfg, "paths.experiment_csv_dir", "outputs/experiments")
    ensure_dir(experiment_csv_dir)

    # ---- 指标采集器 ----
    tag = cfg.get("experiment", {}).get("tag", "model")
    collector = MetricsCollector(output_dir=experiment_csv_dir, experiment_tag=tag)
    print(f"[info] metrics CSV: {collector.train_csv_path}")

    # ---- 训练 ----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        cfg=cfg,
        device=device,
        output_dir=Path(str(ckpt_dir)),
        scheduler=scheduler,
        collector=collector,
    )
    epochs = args.epochs or train_cfg.get("epochs", 5)
    summary = trainer.fit(epochs=epochs)

    print(f"[info] training summary: {summary}")
    return 0


def _limit_dataset(ds, n: int):
    """限制数据集大小为 n，用于快速测试。"""
    total = len(ds)
    n = min(n, total)
    ds.ids = ds.ids[:n]
    print(f"[info] dataset limited from {total} to {n} samples")


def _limit_dataset_cfg(cfg: dict, n: int):
    """用 max_samples 覆盖配置中的 epochs 使训练快速完成。"""
    cfg.setdefault("train", {})["epochs"] = min(cfg.get("train", {}).get("epochs", 5), 3)
    if cfg.get("train", {}).get("batch_size", 16) > n:
        cfg["train"]["batch_size"] = min(4, n)


if __name__ == "__main__":
    raise SystemExit(main())
