"""Checkpoint 管理器 —— 定期保存 + 最佳 mAP + 训练恢复。

依据：docxs/RESEARCH.md §3 训练中断恢复与 Checkpoint 策略
    每个 epoch 结束后保存完整状态（model+optimizer+scheduler+epoch），
    确保训练中断后可从最近 epoch 无缝恢复。

用法::

    ckpt_mgr = CheckpointManager(output_dir, tag="baseline_plus_lca")

    # 每 epoch 保存
    ckpt_mgr.save_latest(model, optimizer, scheduler, epoch, metrics)

    # mAP 提升时保存 best
    if current_map > best_map:
        ckpt_mgr.save_best(model, epoch, {"mAP50": current_map})

    # 训练恢复
    state = ckpt_mgr.load_latest(model, optimizer, scheduler)
    start_epoch = state["epoch"] + 1
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CheckpointManager:
    """管理训练 checkpoint 的保存与恢复。

    Args:
        output_dir: checkpoint 存放目录。
        tag: 实验标识（如 "baseline_plus_lca"），用于命名文件。
        keep_interval: 保留周期性快照的间隔（epoch 数），-1 表示不保留。
        keep_best_only: 只保留 best checkpoint（不保留 latest 和 interval）。
        max_snapshots: 最多保留几个周期性快照。
    """

    def __init__(
        self,
        output_dir: str | Path,
        tag: str = "model",
        keep_interval: int = 10,
        keep_best_only: bool = False,
        max_snapshots: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.keep_interval = keep_interval
        self.keep_best_only = keep_best_only
        self.max_snapshots = max_snapshots

    # ── 路径属性 ──────────────────────────────────────

    @property
    def latest_path(self) -> Path:
        return self.output_dir / f"{self.tag}_latest.pt"

    @property
    def best_path(self) -> Path:
        return self.output_dir / f"{self.tag}_best.pt"

    def snapshot_path(self, epoch: int) -> Path:
        return self.output_dir / f"{self.tag}_epoch_{epoch:04d}.pt"

    # ── 保存 ──────────────────────────────────────────

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int = 0,
        metrics: dict[str, Any] | None = None,
        path: Path | None = None,
    ) -> Path:
        """保存完整 checkpoint（原子写入：先写临时文件再 rename）。"""
        ckpt: dict[str, Any] = {
            "model_state_dict": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
            "epoch": epoch,
            "tag": self.tag,
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        if metrics:
            ckpt["metrics"] = metrics

        save_path = path or self.latest_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 原子写入（Windows: 必须关闭 fd 再 move）
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".pt", prefix="ckpt_", dir=str(save_path.parent)
        )
        try:
            torch.save(ckpt, tmp_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        finally:
            import os
            os.close(tmp_fd)
        # Windows: 先删目标再 rename
        if save_path.exists():
            save_path.unlink()
        shutil.move(tmp_path, str(save_path))

        return save_path

    def save_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        epoch: int,
        metrics: dict[str, Any] | None = None,
    ) -> Path:
        """保存 latest checkpoint（每 epoch 覆盖）。"""
        if self.keep_best_only:
            return self.latest_path
        return self.save(model, optimizer, scheduler, epoch, metrics, self.latest_path)

    def save_best(
        self,
        model: nn.Module,
        epoch: int,
        metrics: dict[str, Any],
    ) -> Path:
        """保存 best checkpoint（mAP 提升时调用）。"""
        return self.save(
            model, optimizer=None, scheduler=None,
            epoch=epoch, metrics=metrics, path=self.best_path,
        )

    def save_snapshot(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        epoch: int,
        metrics: dict[str, Any] | None = None,
    ) -> Path | None:
        """保存周期性快照（每 keep_interval epoch）。自动清理旧快照。"""
        if self.keep_interval <= 0 or self.keep_best_only:
            return None
        if epoch % self.keep_interval != 0:
            return None

        path = self.snapshot_path(epoch)
        result = self.save(model, optimizer, scheduler, epoch, metrics, path)

        # 清理旧快照
        self._cleanup_snapshots(epoch)

        return result

    def _cleanup_snapshots(self, current_epoch: int) -> None:
        """仅保留最近 max_snapshots 个快照。"""
        snapshots = sorted(
            self.output_dir.glob(f"{self.tag}_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        # 保留当前 epoch 的快照
        keep = {self.snapshot_path(current_epoch).name}
        excess = [p for p in snapshots if p.name not in keep]
        while len(excess) >= self.max_snapshots:
            oldest = excess.pop(0)
            oldest.unlink(missing_ok=True)

    # ── 恢复 ──────────────────────────────────────────

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        path: Path | None = None,
    ) -> dict[str, Any]:
        """从指定路径恢复训练状态。

        Returns:
            dict with keys: model_state_dict, epoch, optimizer_state_dict,
            scheduler_state_dict, metrics, tag.
        """
        load_path = path or self.latest_path
        if not load_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {load_path}")

        ckpt = torch.load(str(load_path), map_location="cpu", weights_only=False)

        # 恢复模型权重
        model.load_state_dict(ckpt["model_state_dict"])

        # 恢复优化器
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # 恢复调度器
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (KeyError, TypeError, AttributeError):
                print("[warn] scheduler state_dict 加载失败，将从头开始调度")

        return ckpt

    def load_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
    ) -> dict[str, Any]:
        """从 latest checkpoint 恢复训练。"""
        return self.load(model, optimizer, scheduler, self.latest_path)

    def load_best_weights(self, model: nn.Module) -> dict[str, Any]:
        """仅加载 best checkpoint 的模型权重（用于最终评估）。"""
        return self.load(model, path=self.best_path)

    def has_latest(self) -> bool:
        return self.latest_path.exists()

    def has_best(self) -> bool:
        return self.best_path.exists()
