"""学习率调度器 —— Linear Warmup + Cosine Annealing（按 iteration 步进）。

依据：docs/TRAIN-1.md + docxs/RESEARCH.md
    现代化目标检测训练的事实标准：Warmup (5-10% 总步数) + Cosine Annealing。

实现选择：
    PyTorch >= 2.0 推荐 SequentialLR(LinearLR, CosineAnnealingLR)，简洁且经充分测试。
    每个 optimizer.step() 之后调用 scheduler.step()（即按 iteration 步进）。

用法::

    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch, epochs)
    for epoch in range(epochs):
        for batch in train_loader:
            ...
            optimizer.step()
            scheduler.step()   # ← 每个 iteration 步进
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR, LinearLR, CosineAnnealingLR, SequentialLR


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """构建 Linear Warmup + Cosine Annealing 调度器。

    Args:
        optimizer: 优化器实例。
        cfg: 完整配置；读取 lr_scheduler / stage2_joint_training 段。
        steps_per_epoch: 每 epoch 的 iteration 数。
        total_epochs: 阶段二总 epoch 数（用于 T_max 计算）。

    Returns:
        LRScheduler：按 iteration 步进，自动处理 warmup → cosine 过渡。
    """
    sched_cfg: dict = cfg.get("lr_scheduler", {})
    warmup_iters: int = int(sched_cfg.get("warmup_iters", 1000))
    warmup_start_factor: float = float(sched_cfg.get("warmup_start_factor", 0.001))

    # 若 warmup_iters 用负数表示百分比（如 -0.1 = 10%），则从总步数计算
    if warmup_iters < 0:
        total_iters = total_epochs * steps_per_epoch
        warmup_iters = int(abs(warmup_iters) * total_iters)

    # 确保 warmup 不超过总步数
    total_iters = total_epochs * steps_per_epoch
    warmup_iters = min(warmup_iters, max(1, total_iters // 2))

    # Cosine 阶段步数
    cosine_iters = total_iters - warmup_iters

    # Linear warmup: 从 warmup_start_factor * base_lr → base_lr
    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_iters,
    )

    # Cosine annealing: base_lr → eta_min
    eta_min_factor: float = float(sched_cfg.get("eta_min_factor", 0.0))
    # 对每组参数：eta_min = base_lr * eta_min_factor
    # CosineAnnealingLR 的 eta_min 是绝对值，需为每组设置不同值
    # 使用 LambdaLR 包装来支持多 LR 组的不同 eta_min

    anneal = CosineAnnealingLR(
        optimizer,
        T_max=cosine_iters,
        eta_min=0.0,  # 绝对值不适用多 LR 组；用 SequentialLR 的 milestone 机制
    )

    # 注意：CosineAnnealingLR 的 eta_min 对所有 param_groups 相同，不适用 LLRD
    # 改用自定义 LambdaLR 实现，为每组独立计算 cosine 衰减到各自的 eta_min

    # 重建：使用统一 LambdaLR 实现，每组独立衰减
    # 先删除刚创建的实例引用，用自定义替换
    scheduler = _WarmupCosineLR(
        optimizer,
        warmup_iters=warmup_iters,
        total_iters=total_iters,
        warmup_start_factor=warmup_start_factor,
        eta_min_factor=eta_min_factor,
    )

    return scheduler


def build_scheduler_from_total_iters(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int = 1000,
    total_iters: int = 100000,
    warmup_start_factor: float = 0.001,
    eta_min_factor: float = 0.0,
) -> _WarmupCosineLR:
    """直接参数构建调度器（测试友好）。"""
    return _WarmupCosineLR(
        optimizer,
        warmup_iters=warmup_iters,
        total_iters=total_iters,
        warmup_start_factor=warmup_start_factor,
        eta_min_factor=eta_min_factor,
    )


class _WarmupCosineLR(LambdaLR):
    """Linear Warmup + Cosine Annealing，支持多 LR 组独立衰减。

    每组参数从其 initial_lr 开始：
        - warmup 阶段：linear 从 initial_lr * warmup_start_factor → initial_lr
        - cosine 阶段：cosine 从 initial_lr → initial_lr * eta_min_factor
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        total_iters: int,
        warmup_start_factor: float = 0.001,
        eta_min_factor: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.warmup_start_factor = warmup_start_factor
        self.eta_min_factor = eta_min_factor
        self._current_iter = 0

        def lr_lambda(step: int) -> float:
            """返回当前步的 LR 乘数（相对 initial_lr）。"""
            if step < warmup_iters:
                # Linear warmup: start_factor → 1.0
                alpha = step / max(1, warmup_iters)
                return warmup_start_factor + (1.0 - warmup_start_factor) * alpha
            else:
                # Cosine annealing: 1.0 → eta_min_factor
                progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return eta_min_factor + (1.0 - eta_min_factor) * cosine

        super().__init__(optimizer, lr_lambda, last_epoch)

    def get_last_lr(self) -> list[float]:
        """返回当前所有参数组的实际 LR。"""
        return [group["lr"] for group in self.optimizer.param_groups]
