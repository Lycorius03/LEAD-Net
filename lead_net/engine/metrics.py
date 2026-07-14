"""训练指标采集器 —— CSV 增量写入，零跨模块依赖。

采集轨道：
    - train_metrics.csv：每 epoch 一行（loss/mAP/lr/时间/GPU）
    - per_class_ap.csv：每 eval_interval 轮一次（各类别 AP）

工业标准参照：
    - YOLO results.csv：epoch / train+val loss / mAP / lr
    - COCO eval：mAP@0.5 / mAP@0.5:0.95 / mAP@0.75 / per-class AP
    - 诊断信号：train-val loss gap（过拟合）、lr vs loss 相关性

设计原则：
    - 零跨模块依赖（仅依赖标准库 csv/pathlib/datetime）
    - 增量写入（每 epoch append，不积攒内存，训练崩溃时已写数据不丢）
    - 可配置字段（通过 train_metrics_fields 控制列集合）
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# 默认 train_metrics.csv 列集合（工业标准全集）
_DEFAULT_TRAIN_FIELDS = [
    "epoch",
    "lr",
    "train/cls_loss",
    "train/loc_loss",
    "train/loss",
    "val/cls_loss",
    "val/loc_loss",
    "val/loss",
    "val/mAP@0.5",
    "val/mAP@0.5:0.95",
    "val/mAP@0.75",
    "epoch_time_s",
    "samples_per_sec",
    "gpu_memory_mb",
    "timestamp",
]

_DEFAULT_PER_CLASS_FIELDS = [
    "epoch",
    "class_id",
    "class_name",
    "AP@0.5",
    "AP@0.5:0.95",
    "AP@0.75",
]


class MetricsCollector:
    """训练指标采集与 CSV 持久化。

    用法::

        collector = MetricsCollector(output_dir, "baseline_no_lca")

        # 每 epoch：
        collector.log_epoch({
            "epoch": 1,
            "lr": 0.01,
            "train/cls_loss": 2.3,
            "train/loc_loss": 1.2,
            ...
        })

        # 每 eval_interval：
        collector.log_per_class(5, [
            {"class_id": 0, "class_name": "person", "AP@0.5": 0.45, ...},
            ...
        ])

    Args:
        output_dir: CSV 输出目录（通常为 outputs/experiments/）。
        experiment_tag: 实验标识（如 "baseline_no_lca"），用于命名 CSV 文件。
        train_fields: 自定义 train_metrics.csv 列集合（None=默认全集）。
        per_class_fields: 自定义 per_class_ap.csv 列集合。
    """

    def __init__(
        self,
        output_dir: str | Path,
        experiment_tag: str,
        train_fields: list[str] | None = None,
        per_class_fields: list[str] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tag = experiment_tag

        self._train_path = self.output_dir / f"{experiment_tag}_train_metrics.csv"
        self._per_class_path = self.output_dir / f"{experiment_tag}_per_class_ap.csv"

        self._train_fields = train_fields or _DEFAULT_TRAIN_FIELDS
        self._per_class_fields = per_class_fields or _DEFAULT_PER_CLASS_FIELDS

        self._train_written_header = self._train_path.exists()
        self._per_class_written_header = self._per_class_path.exists()

    # ---- public API ----

    def log_epoch(self, metrics: dict[str, Any]) -> None:
        """写入一行 epoch 指标到 train_metrics.csv。

        缺失字段自动填 None；集合外字段忽略（不崩但也不写入）。
        """
        row = {f: metrics.get(f) for f in self._train_fields}
        row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

        if not self._train_written_header:
            self._train_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._train_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._train_fields)
                writer.writeheader()
            self._train_written_header = True

        with open(self._train_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._train_fields)
            writer.writerow(row)

    def log_per_class(self, epoch: int, records: list[dict[str, Any]]) -> None:
        """批量写入 per-class AP 到 per_class_ap.csv。

        Args:
            epoch: 当前 epoch 编号。
            records: list of dict，每 dict 含 class_id/class_name/各 AP 字段。
        """
        if not self._per_class_written_header:
            self._per_class_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._per_class_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._per_class_fields)
                writer.writeheader()
            self._per_class_written_header = True

        with open(self._per_class_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._per_class_fields)
            for rec in records:
                rec["epoch"] = epoch
                row = {k: rec.get(k) for k in self._per_class_fields}
                writer.writerow(row)

    @property
    def train_csv_path(self) -> Path:
        return self._train_path

    @property
    def per_class_csv_path(self) -> Path:
        return self._per_class_path

    # ---- static helpers ----

    @staticmethod
    def gpu_memory_mb(device: Any = None) -> float | None:
        """获取当前 GPU 已分配显存 (MB)，失败返回 None。"""
        try:
            import torch
            return torch.cuda.memory_allocated(device) / (1024 * 1024)
        except Exception:
            return None

    @staticmethod
    def timestamp_iso() -> str:
        return datetime.now().isoformat(timespec="seconds")
