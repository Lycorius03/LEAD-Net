"""云端训练器 —— 两阶段 LLRD + EMA + Auto Batch + iteration warmup。

依据：docs/TRAIN-1.md（最终定稿 v2）

训练流程：
    阶段一：冻结 Backbone → 仅训练 LCA + Head（3-5 epoch，动态判断解冻时机）
    阶段二：解冻 Backbone → LLRD 联合训练 120 epoch（不 Early Stop）
    全程：EMA 参数副本 + iteration warmup + auto batch + cosine 调度
"""

from __future__ import annotations

import copy
import math
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
    from lead_net.engine.ema import ModelEMA


class Trainer:
    """云端训练器（支持两阶段 + LLRD + EMA + Auto Batch）。

    用法::

        trainer = Trainer(model, train_loader, val_loader, criterion, cfg, device, ...)
        summary = trainer.fit()  # 自动执行两阶段训练
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        criterion: MultiBoxLoss,
        cfg: dict,
        device: torch.device | None = None,
        output_dir: Path | None = None,
        collector: MetricsCollector | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/checkpoints")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collector = collector
        self.model.to(self.device)

        train_cfg = cfg.get("training", {})
        self._use_amp = train_cfg.get("amp", False)
        self._use_ema = train_cfg.get("ema", False)
        self._ema_decay = train_cfg.get("ema_decay", 0.9998)
        self._grad_clip = train_cfg.get("grad_clip_norm", 10.0)
        self._scaler = torch.amp.GradScaler("cuda") if self._use_amp else None

        # EMA
        self._ema: ModelEMA | None = None
        from lead_net.engine.ema import ModelEMA
        self._ema = ModelEMA(self.model, decay=self._ema_decay) if self._use_ema else None

        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any = None
        self._warmup_iters: int = 0
        self._total_iters: int = 0

    # ---- fit (main) ----

    def fit(self) -> dict[str, Any]:
        """执行完整两阶段训练。返回 training summary。"""
        cfg = self.cfg
        tag = cfg.get("experiment", {}).get("tag", "model")
        train_cfg = cfg.get("training", {})
        eval_interval = cfg.get("eval", {}).get("eval_interval", 5)

        bs = train_cfg.get("batch_size", 16)
        if isinstance(bs, str) and bs.lower() == "auto":
            from lead_net.engine.auto_batch import auto_batch_size
            bs = auto_batch_size(self.model, self.train_loader, cfg, self.device)
            self.train_loader = self._rebuild_loader(bs)
            print(f"[train] auto batch → {bs}")
        print(f"[train] batch_size={bs}")

        total_start = time.time()
        best_mAP50 = -1.0
        best_epoch = -1
        best_state: dict | None = None

        # ---- 阶段一：冻结 Backbone ----
        stage1_cfg = cfg.get("stage1_freeze_backbone", {})
        if stage1_cfg.get("enabled", False):
            print("\n=== Stage 1: Freeze Backbone ===")
            self._setup_optimizer(freeze_backbone=True)
            print(f"[train] DataLoader ready, {len(self.train_loader)} batches/epoch "
                  f"(first batch loads {self.train_loader.batch_size} images, may take 1-2 min)...")
            s1_epochs = self._run_stage1(stage1_cfg, eval_interval)
            print(f"[train] stage1 completed after {s1_epochs} epochs")
        else:
            self._setup_optimizer(freeze_backbone=False)

        # ---- 阶段二：LLRD 联合训练 ----
        stage2_cfg = cfg.get("stage2_joint_training", {})
        s2_epochs = stage2_cfg.get("epochs", 120)
        patience = stage2_cfg.get("patience", 0) if stage2_cfg.get("early_stopping", False) else 0
        patience_counter = 0
        print(f"\n=== Stage 2: Joint Training ({s2_epochs} epochs) ===")
        if patience > 0:
            print(f"[train] early stopping: patience={patience} (on mAP@0.5)")

        for epoch in range(1, s2_epochs + 1):
            t0 = time.time()
            lrs = [f"{pg['lr']:.2e}" for pg in self._optimizer.param_groups]
            print(f"\nEpoch {epoch}/{s2_epochs}  lr={lrs}", flush=True)
            train_m = self._train_one_epoch(epoch)
            train_loss = train_m["cls_loss"] + train_m["loc_loss"]

            val_m = self._validate_one_epoch()
            val_loss = (val_m["cls_loss"] + val_m["loc_loss"]) if val_m else None

            eval_metrics = None
            if (epoch % eval_interval == 0 or epoch == s2_epochs) and self.val_loader:
                print(f"running COCO mAP evaluation...", flush=True)
                eval_metrics = self._run_eval()

            epoch_time = time.time() - t0
            lr = self._optimizer.param_groups[0]["lr"]
            samples_per_sec = len(self.train_loader.dataset) / max(epoch_time, 1e-8)
            gpu_mem = _gpu_memory_mb(self.device)

            # print
            val_s = f"val_loss: {val_loss:.4f}  val_cls: {val_m['cls_loss']:.4f}  val_loc: {val_m['loc_loss']:.4f}" if val_m else ""
            map_s = ""
            if eval_metrics:
                map_s = f"mAP@0.5: {eval_metrics.get('mAP@0.5',0):.4f}  mAP@0.5:0.95: {eval_metrics.get('mAP@0.5:0.95',0):.4f}  mAP@0.75: {eval_metrics.get('mAP@0.75',0):.4f}"
            print(f"train_loss: {train_loss:.4f}  cls_loss: {train_m['cls_loss']:.4f}  loc_loss: {train_m['loc_loss']:.4f}  "
                  f"grad: {train_m['grad_norm']:.2f}", flush=True)
            if val_s:
                print(val_s, flush=True)
            if map_s:
                print(map_s, flush=True)
            print(f"lr: {lr:.2e}  {epoch_time:.0f}s  {samples_per_sec:.0f} img/s  GPU: {gpu_mem:.0f}MB", flush=True)

            # log CSV
            if self.collector:
                self.collector.log_epoch({
                    "epoch": epoch, "lr": lr,
                    "train/cls_loss": train_m["cls_loss"], "train/loc_loss": train_m["loc_loss"],
                    "train/loss": train_loss,
                    "val/cls_loss": val_m["cls_loss"] if val_m else None,
                    "val/loc_loss": val_m["loc_loss"] if val_m else None, "val/loss": val_loss,
                    "val/mAP@0.5": eval_metrics.get("mAP@0.5") if eval_metrics else None,
                    "val/mAP@0.5:0.95": eval_metrics.get("mAP@0.5:0.95") if eval_metrics else None,
                    "val/mAP@0.75": eval_metrics.get("mAP@0.75") if eval_metrics else None,
                    "epoch_time_s": round(epoch_time, 1), "samples_per_sec": round(samples_per_sec, 1),
                    "gpu_memory_mb": round(gpu_mem, 1) if gpu_mem else None,
                    "grad_norm": train_m.get("grad_norm"),
                    "gpu_model": torch.cuda.get_device_name(self.device) if torch.cuda.is_available() else "CPU",
                    "seed": cfg.get("seed"),
                })
                if eval_metrics and eval_metrics.get("per_class"):
                    self.collector.log_per_class(epoch, eval_metrics["per_class"])

            # best checkpoint
            current_map = eval_metrics.get("mAP@0.5", -1) if eval_metrics else -1
            if current_map > best_mAP50:
                best_mAP50 = current_map
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            elif eval_metrics is not None and patience > 0:
                patience_counter += 1
                print(f"[train] patience: {patience_counter}/{patience} (best mAP@0.5={best_mAP50:.4f} at epoch {best_epoch})", flush=True)
                if patience_counter >= patience:
                    print(f"[train] early stopping triggered at epoch {epoch}", flush=True)
                    break

        # ---- 训练结束 ----
        if cfg.get("checkpoint", {}).get("restore_best_at_end", True) and best_state:
            self.model.load_state_dict(best_state)
            print(f"[train] restored best model (epoch {best_epoch}, mAP@0.5={best_mAP50:.4f})")

        ckpt_path = self.output_dir / f"{tag}.pt"
        torch.save({"model_state_dict": self.model.state_dict(), "epoch": s2_epochs, "tag": tag}, str(ckpt_path))
        _update_last_link(self.output_dir, ckpt_path)

        if best_state:
            best_path = self.output_dir / f"{tag}_best.pt"
            torch.save({"model_state_dict": best_state, "epoch": best_epoch, "tag": tag, "mAP50": best_mAP50}, str(best_path))

        total_time = time.time() - total_start
        summary = {"best_epoch": best_epoch, "best_mAP50": best_mAP50, "total_epochs": s2_epochs, "total_time_s": round(total_time, 1)}
        print(f"[train] done | best: e{best_epoch} mAP@0.5={best_mAP50:.4f} | {total_time:.0f}s")
        return summary

    # ---- stage 1 ----

    def _run_stage1(self, stage1_cfg: dict, eval_interval: int) -> int:
        max_eps = stage1_cfg.get("max_epochs", 5)
        threshold = stage1_cfg.get("loss_change_threshold", 0.05)
        prev_loss = float("inf")
        stable_count = 0

        for epoch in range(1, max_eps + 1):
            t0 = time.time()
            lrs = [f"{pg['lr']:.2e}" for pg in self._optimizer.param_groups]
            print(f"\nEpoch {epoch}/{max_eps} (freeze backbone)  lr={lrs}", flush=True)
            train_m = self._train_one_epoch(epoch)
            loss = train_m["cls_loss"] + train_m["loc_loss"]
            elapsed = time.time() - t0
            change = abs(prev_loss - loss) / max(abs(prev_loss), 1e-8)

            if self.val_loader:
                val_m = self._validate_one_epoch()
                val_loss = val_m["cls_loss"] + val_m["loc_loss"] if val_m else 0.0
            else:
                val_m, val_loss = None, 0.0

            val_s = f"val_loss: {val_loss:.4f}  val_cls: {val_m['cls_loss']:.4f}  val_loc: {val_m['loc_loss']:.4f}" if val_m else ""
            print(f"train_loss: {loss:.4f}  cls_loss: {train_m['cls_loss']:.4f}  loc_loss: {train_m['loc_loss']:.4f}  "
                  f"grad: {train_m['grad_norm']:.2f}  Δ: {change:.4f}  {elapsed:.0f}s", flush=True)
            if val_s:
                print(val_s, flush=True)

            if change < threshold:
                stable_count += 1
            else:
                stable_count = 0
            prev_loss = loss

            if stable_count >= 2:
                print(f"loss stabilized (Δ<{threshold} for 2 epochs) → unfreezing", flush=True)
                return epoch

        return max_eps

    # ---- single epoch ----

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_cls, total_loc, total_grad, n = 0.0, 0.0, 0.0, 0
        warmup_done = False

        total_batches = len(self.train_loader)
        t_batch_start = time.time()
        for bi, batch in enumerate(self.train_loader):
            # iteration warmup
            global_step = (epoch - 1) * len(self.train_loader) + bi
            if global_step < self._warmup_iters and not warmup_done:
                scale = float(global_step + 1) / max(self._warmup_iters, 1)
                for pg in self._optimizer.param_groups:
                    pg["lr"] = pg.get("initial_lr", pg["lr"]) * scale
            elif not warmup_done and self._warmup_iters > 0:
                warmup_done = True

            images = batch["image"].to(self.device)
            gt_boxes, gt_labels = batch["boxes"], batch["labels"]

            if self._scaler:
                with torch.amp.autocast("cuda"):
                    cls_pred, loc_pred = self.model(images)
                    default_boxes = self.model.head.all_default_boxes(self.device)
                    cls_loss, loc_loss = self.criterion(cls_pred, loc_pred, default_boxes, gt_boxes, gt_labels)
                    loss = cls_loss + loc_loss
                self._scaler.scale(loss).backward()
            else:
                cls_pred, loc_pred = self.model(images)
                default_boxes = self.model.head.all_default_boxes(self.device)
                cls_loss, loc_loss = self.criterion(cls_pred, loc_pred, default_boxes, gt_boxes, gt_labels)
                loss = cls_loss + loc_loss
                self._optimizer.zero_grad()
                loss.backward()

            total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip).item()
            if self._scaler:
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                self._optimizer.step()
                self._optimizer.zero_grad()

            if self._ema:
                self._ema.update()

            total_cls += cls_loss.item()
            total_loc += loc_loss.item()
            total_grad += total_norm  # type: ignore[assignment]
            n += 1

            # progress: every 20 batches + first + last
            if bi == 0 or (bi + 1) % 20 == 0 or bi == total_batches - 1:
                avg_cls = total_cls / max(n, 1)
                avg_loc = total_loc / max(n, 1)
                elapsed = time.time() - t_batch_start
                lr = self._optimizer.param_groups[0]["lr"]
                print(f"batch {bi+1:4d}/{total_batches}, "
                      f"cls: {avg_cls:.4f}  loc: {avg_loc:.4f}  "
                      f"loss: {avg_cls+avg_loc:.4f}  "
                      f"grad: {total_grad/max(n,1):.2f}  "
                      f"lr: {lr:.2e}  {elapsed:.0f}s", flush=True)

        if self._scheduler:
            self._scheduler.step()

        return {"cls_loss": total_cls / max(n, 1), "loc_loss": total_loc / max(n, 1), "grad_norm": total_grad / max(n, 1)}

    @torch.no_grad()
    def _validate_one_epoch(self) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        if self._ema:
            self._ema.apply()
        self.model.eval()
        total_cls, total_loc, n = 0.0, 0.0, 0
        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            cls_pred, loc_pred = self.model(images)
            dboxes = self.model.head.all_default_boxes(self.device)
            cl, ll = self.criterion(cls_pred, loc_pred, dboxes, batch["boxes"], batch["labels"])
            total_cls += cl.item()
            total_loc += ll.item()
            n += 1
        if self._ema:
            self._ema.restore()
        return {"cls_loss": total_cls / max(n, 1), "loc_loss": total_loc / max(n, 1)}

    def _run_eval(self) -> dict | None:
        from lead_net.engine.evaluator import Evaluator
        if self._ema:
            self._ema.apply()
        evaluator = Evaluator(self.model, self.val_loader, self.cfg, self.device)
        result = evaluator.evaluate()
        if self._ema:
            self._ema.restore()
        return result

    def _setup_optimizer(self, freeze_backbone: bool = False) -> None:
        from lead_net.engine.llrd import build_llrd_param_groups
        groups = build_llrd_param_groups(self.model, self.cfg, freeze_backbone=freeze_backbone)
        opt_cfg = self.cfg.get("optimizer", {})
        self._optimizer = torch.optim.SGD(
            groups, lr=0.001, momentum=opt_cfg.get("momentum", 0.9),
            nesterov=opt_cfg.get("nesterov", True),
        )
        for pg in self._optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

        s2 = self.cfg.get("stage2_joint_training", {})
        total_epochs = s2.get("epochs", 120)
        self._warmup_iters = self.cfg.get("lr_scheduler", {}).get("warmup_iters", 1000)
        self._total_iters = total_epochs * len(self.train_loader)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self._total_iters)

    def _rebuild_loader(self, batch_size: int) -> DataLoader:
        from lead_net.data import build_dataloader
        import sys
        nw = 0 if sys.platform == "win32" else (self.cfg.get("training") or self.cfg.get("train", {})).get("num_workers", 4)
        return build_dataloader(self.cfg, split="train", batch_size=batch_size, num_workers=nw)


def _gpu_memory_mb(device) -> float | None:
    try:
        return torch.cuda.memory_allocated(device) / (1024 * 1024)
    except Exception:
        return None


def _update_last_link(out_dir: Path, ckpt_path: Path) -> None:
    last_path = out_dir / "last.pt"
    try:
        if last_path.is_symlink() or last_path.exists():
            last_path.unlink()
        last_path.symlink_to(ckpt_path.name)
    except (OSError, NotImplementedError):
        shutil.copy2(ckpt_path, last_path)


# module-level import (avoid circular)
from lead_net.engine.metrics import MetricsCollector  # noqa: E402
