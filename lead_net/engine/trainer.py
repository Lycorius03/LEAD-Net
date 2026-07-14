"""训练器 —— 训练循环 + 验证 + 指标采集。

工业标准参照：
    - 每 epoch 记录 train loss + val loss（诊断过拟合）
    - 每 eval_interval 跑完整 COCO mAP 评估
    - 维护 best checkpoint（基于 val mAP@0.5）
    - 增量 CSV 写入（不积攒内存）
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from lead_net.models.loss import MultiBoxLoss
    from lead_net.engine.metrics import MetricsCollector


class Trainer:
    """LEAD-Net 训练器（训练 + 验证 + 指标采集）。

    每 epoch：
        1. train_one_epoch → 返回 train loss
        2. 若有 val_loader：validate_one_epoch → 返回 val loss
        3. 若 epoch % eval_interval == 0：完整 COCO mAP 评估
        4. 所有指标写入 MetricsCollector CSV
        5. 维护 best checkpoint（最佳 val mAP@0.5）
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        optimizer: torch.optim.Optimizer,
        criterion: MultiBoxLoss,
        cfg: dict,
        device: torch.device | None = None,
        output_dir: Path | None = None,
        scheduler: Any = None,
        collector: MetricsCollector | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/checkpoints")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = scheduler
        self.collector = collector
        self.model.to(self.device)

    # ---- train one epoch ----

    def train_one_epoch(self) -> dict[str, float]:
        """训练一个 epoch，返回 avg cls_loss / loc_loss / loss。"""
        self.model.train()
        total_cls = 0.0
        total_loc = 0.0
        n = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            gt_boxes = batch["boxes"]
            gt_labels = batch["labels"]

            cls_pred, loc_pred = self.model(images)
            default_boxes = self.model.head.all_default_boxes(self.device)
            cls_loss, loc_loss = self.criterion(
                cls_pred, loc_pred, default_boxes, gt_boxes, gt_labels,
            )

            loss = cls_loss + loc_loss
            self.optimizer.zero_grad()
            loss.backward()
            grad_clip = self.cfg.get("train", {}).get("grad_clip", None)
            if grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            total_cls += cls_loss.item()
            total_loc += loc_loss.item()
            n += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "cls_loss": total_cls / max(n, 1),
            "loc_loss": total_loc / max(n, 1),
        }

    # ---- validate one epoch (loss only, no mAP) ----

    @torch.no_grad()
    def validate_one_epoch(self) -> dict[str, float] | None:
        """计算验证集 loss（不跑 NMS/mAP）。无 val_loader 则返回 None。"""
        if self.val_loader is None:
            return None
        self.model.eval()
        total_cls = 0.0
        total_loc = 0.0
        n = 0
        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            gt_boxes = batch["boxes"]
            gt_labels = batch["labels"]

            cls_pred, loc_pred = self.model(images)
            default_boxes = self.model.head.all_default_boxes(self.device)
            cls_loss, loc_loss = self.criterion(
                cls_pred, loc_pred, default_boxes, gt_boxes, gt_labels,
            )
            total_cls += cls_loss.item()
            total_loc += loc_loss.item()
            n += 1
        return {
            "cls_loss": total_cls / max(n, 1),
            "loc_loss": total_loc / max(n, 1),
        }

    # ---- fit (main loop) ----

    def fit(self, epochs: int | None = None) -> dict[str, Any]:
        """完整训练循环。

        Returns:
            summary: {"best_epoch", "best_mAP50", "total_epochs", "total_time_s"}
        """
        train_cfg = self.cfg.get("train", {})
        eval_cfg = self.cfg.get("eval", {})
        epochs = epochs or train_cfg.get("epochs", 5)
        eval_interval = eval_cfg.get("eval_interval", 5)
        tag = self.cfg.get("experiment", {}).get("tag", "model")

        best_mAP50 = -1.0
        best_epoch = -1
        total_start = time.time()
        train_samples = len(self.train_loader.dataset)

        print(f"[train] epochs={epochs}  train_samples={train_samples}  "
              f"val_samples={len(self.val_loader.dataset) if self.val_loader else 0}  "
              f"device={self.device}  eval_interval={eval_interval}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ---- train ----
            train_m = self.train_one_epoch()
            train_loss = train_m["cls_loss"] + train_m["loc_loss"]

            # ---- validate (loss) ----
            val_m = self.validate_one_epoch()
            val_loss = (val_m["cls_loss"] + val_m["loc_loss"]) if val_m else None

            # ---- eval (mAP, every eval_interval) ----
            eval_metrics = None
            if (epoch % eval_interval == 0 or epoch == epochs) and self.val_loader is not None:
                eval_metrics = self._run_eval()

            epoch_time = time.time() - t0
            samples_per_sec = train_samples / max(epoch_time, 1e-8)
            lr = self.optimizer.param_groups[0]["lr"]
            gpu_mem = MetricsCollector.gpu_memory_mb(self.device)

            # ---- print ----
            val_str = ""
            if val_loss is not None:
                val_str = (f"val_loss={val_loss:.4f} "
                           f"(cls={val_m['cls_loss']:.4f} loc={val_m['loc_loss']:.4f}) ")
            map_str = ""
            if eval_metrics:
                map_str = (f"mAP@0.5={eval_metrics.get('mAP@0.5', 0):.4f} "
                           f"mAP@0.5:0.95={eval_metrics.get('mAP@0.5:0.95', 0):.4f} ")
            print(
                f"[train] epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} "
                f"(cls={train_m['cls_loss']:.4f} loc={train_m['loc_loss']:.4f}) | "
                f"{val_str}| "
                f"{map_str}| "
                f"lr={lr:.6f} | {epoch_time:.1f}s | {samples_per_sec:.0f} img/s"
            )

            # ---- log to CSV ----
            if self.collector is not None:
                row = {
                    "epoch": epoch,
                    "lr": lr,
                    "train/cls_loss": train_m["cls_loss"],
                    "train/loc_loss": train_m["loc_loss"],
                    "train/loss": train_loss,
                    "val/cls_loss": val_m["cls_loss"] if val_m else None,
                    "val/loc_loss": val_m["loc_loss"] if val_m else None,
                    "val/loss": val_loss,
                    "val/mAP@0.5": eval_metrics.get("mAP@0.5") if eval_metrics else None,
                    "val/mAP@0.5:0.95": eval_metrics.get("mAP@0.5:0.95") if eval_metrics else None,
                    "val/mAP@0.75": eval_metrics.get("mAP@0.75") if eval_metrics else None,
                    "epoch_time_s": round(epoch_time, 1),
                    "samples_per_sec": round(samples_per_sec, 1),
                    "gpu_memory_mb": round(gpu_mem, 1) if gpu_mem else None,
                }
                self.collector.log_epoch(row)

                # per-class AP
                if eval_metrics and eval_metrics.get("per_class"):
                    self.collector.log_per_class(epoch, eval_metrics["per_class"])

            # ---- best checkpoint ----
            current_map = eval_metrics.get("mAP@0.5", -1) if eval_metrics else -1
            if current_map > best_mAP50:
                best_mAP50 = current_map
                best_epoch = epoch
                best_path = self.output_dir / f"{tag}_best.pth"
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "tag": tag,
                    "mAP50": best_mAP50,
                }, str(best_path))
                print(f"[train] best checkpoint (mAP@0.5={best_mAP50:.4f}) -> {best_path.name}")

        # ---- final checkpoint ----
        total_time = time.time() - total_start
        ckpt_path = self.output_dir / f"{tag}.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epochs,
            "tag": tag,
        }, str(ckpt_path))
        print(f"[train] final checkpoint saved to {ckpt_path}")
        _update_last_link(self.output_dir, ckpt_path)

        summary = {
            "best_epoch": best_epoch,
            "best_mAP50": best_mAP50,
            "total_epochs": epochs,
            "total_time_s": round(total_time, 1),
        }
        print(f"[train] done | best: epoch={best_epoch} mAP@0.5={best_mAP50:.4f} | "
              f"total={total_time:.0f}s")
        return summary

    # ---- internal ----

    def _run_eval(self) -> dict[str, Any] | None:
        """运行一次完整 COCO 评估，返回详细指标。"""
        from lead_net.engine.evaluator import Evaluator
        evaluator = Evaluator(self.model, self.val_loader, self.cfg, self.device)
        return evaluator.evaluate()


# 静态导入放文件尾，避免 Trainer 初始化时循环依赖
from lead_net.engine.metrics import MetricsCollector


def _update_last_link(out_dir: Path, ckpt_path: Path) -> None:
    """维护 out_dir/last.pth 指向最新 checkpoint。symlink 失败降级为 copy。"""
    last_path = out_dir / "last.pth"
    try:
        if last_path.is_symlink() or last_path.exists():
            last_path.unlink()
        last_path.symlink_to(ckpt_path.name)
        print(f"[train] last.pth -> {ckpt_path.name}")
    except (OSError, NotImplementedError) as e:
        shutil.copy2(ckpt_path, last_path)
        print(f"[train] last.pth (copy, symlink 降级: {type(e).__name__}) -> {ckpt_path.name}")
