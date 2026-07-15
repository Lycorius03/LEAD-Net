"""实验目录管理器 —— 会话分组 + 变体隔离 + 测试分类。

目录结构（一次实验 = 一个 session）:
    outputs/experiments/
    └── exp_001/                          # 实验会话
        ├── session_info.json
        ├── baseline/                     # 变体 A: 无 LCA
        │   ├── train/run_001/            # 训练产物
        │   │   ├── config_snapshot.yaml
        │   │   ├── checkpoints/
        │   │   ├── train_metrics.csv
        │   │   ├── per_class_ap.csv
        │   │   └── model_profile.csv
        │   ├── eval/                     # 训练后评估
        │   │   ├── pr_curve.csv
        │   │   ├── confusion.csv
        │   │   └── per_iou_ap.csv
        │   ├── tracking/                 # 追踪测试
        │   │   ├── with_kalman.csv
        │   │   └── no_kalman.csv
        │   ├── quantization/             # 量化对比
        │   │   └── int8_comparison.csv
        │   └── failure_cases/            # 失败案例
        └── lca/                          # 变体 B: +LCA (结构同上)
            ├── train/run_001/
            ├── eval/
            ├── tracking/
            ├── quantization/
            └── failure_cases/
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentManager:
    """实验目录管理器。

    用法:
        mgr = ExperimentManager("outputs/experiments", "exp_001", "baseline")
        mgr.setup(cfg)
        # → outputs/experiments/exp_001/baseline/train/run_001/
    """

    def __init__(self, base_dir: str | Path, session: str, variant: str):
        self.base_dir = Path(base_dir)
        self.session = session
        self.variant = variant
        self._run_dir: Path | None = None

    # ---- 基础路径 ----
    @property
    def session_dir(self) -> Path:
        return self.base_dir / self.session

    @property
    def variant_dir(self) -> Path:
        return self.session_dir / self.variant

    # ---- 训练 ----
    @property
    def train_dir(self) -> Path:
        return self.variant_dir / "train"

    @property
    def run_dir(self) -> Path:
        if self._run_dir is None:
            run_id = self._next_run_id()
            self._run_dir = self.train_dir / f"run_{run_id:03d}"
        return self._run_dir

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    # ---- 评估 ----
    @property
    def eval_dir(self) -> Path:
        return self.variant_dir / "eval"

    # ---- 追踪 ----
    @property
    def tracking_dir(self) -> Path:
        return self.variant_dir / "tracking"

    # ---- 量化 ----
    @property
    def quantization_dir(self) -> Path:
        return self.variant_dir / "quantization"

    # ---- 失败案例 ----
    @property
    def failure_dir(self) -> Path:
        return self.variant_dir / "failure_cases"

    # ---- 跨变体（全局） ----
    @property
    def cross_dir(self) -> Path:
        return self.session_dir / "cross"

    @property
    def ablation_path(self) -> Path:
        return self.base_dir / "ablation_matrix.csv"

    @staticmethod
    def global_dir(base_dir: str | Path, name: str) -> Path:
        p = Path(base_dir) / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---- setup ----

    def setup(self, cfg: dict | None = None) -> Path:
        """创建完整目录树 + 保存元信息。"""
        dirs = [
            self.run_dir, self.checkpoint_dir,
            self.eval_dir, self.tracking_dir,
            self.quantization_dir, self.failure_dir,
            self.cross_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        self._save_session_info()
        if cfg:
            self._save_config(cfg)
        return self.run_dir

    def _save_session_info(self) -> None:
        variants = set()
        if self.session_dir.exists():
            for d in self.session_dir.iterdir():
                if d.is_dir() and d.name not in (".", "_", "cross"):
                    variants.add(d.name)
        variants.add(self.variant)
        info = {"session": self.session, "created": datetime.now().isoformat(),
                "variants": sorted(variants)}
        with open(self.session_dir / "session_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def _save_config(self, cfg: dict) -> None:
        import yaml
        meta = {"session": self.session, "variant": self.variant,
                "run_id": self.run_dir.name, "timestamp": datetime.now().isoformat()}
        with open(self.run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            yaml.dump({"meta": meta, "config": cfg}, f, allow_unicode=True, default_flow_style=False)

    def _next_run_id(self) -> int:
        if not self.train_dir.exists():
            return 1
        runs = sorted(self.train_dir.glob("run_*"))
        if not runs:
            return 1
        ids = []
        for p in runs:
            try:
                ids.append(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return max(ids) + 1 if ids else 1

    # ---- 静态工具 ----

    @staticmethod
    def next_session(base_dir: str | Path) -> str:
        p = Path(base_dir)
        if not p.exists():
            return "exp_001"
        nums = []
        for d in p.iterdir():
            if d.is_dir() and d.name.startswith("exp_"):
                try:
                    nums.append(int(d.name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        return f"exp_{max(nums)+1:03d}" if nums else "exp_001"

    @staticmethod
    def latest_session(base_dir: str | Path) -> str | None:
        p = Path(base_dir)
        if not p.exists():
            return None
        sessions = sorted(
            d.name for d in p.iterdir()
            if d.is_dir() and d.name.startswith("exp_")
        )
        return sessions[-1] if sessions else None

    @staticmethod
    def for_test(base_dir: str | Path, variant: str,
                 session: str | None = None) -> "ExperimentManager":
        """快捷创建：自动使用最新 session（供 eval/tracking 等测试工具使用）。"""
        s = session or ExperimentManager.latest_session(base_dir) or "exp_001"
        return ExperimentManager(base_dir, s, variant)
