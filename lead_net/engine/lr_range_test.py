"""LR Range Test —— 自动确定最优初始学习率。

基于 Leslie Smith (2015/2018) 的方法:
    "Cyclical Learning Rates for Training Neural Networks"

算法：
    1. 从极小 LR (1e-7) 开始
    2. 每个 batch 指数增加 LR
    3. 记录 (lr, loss) 曲线
    4. 最优初始 LR = loss 下降最快点 / 3 到 /10

用法:
    from lead_net.engine.lr_range_test import LRRangeTest, suggest_lr

    # 方式 1: 手动集成
    tester = LRRangeTest(optimizer, start_lr=1e-7, end_lr=1.0, steps=500)
    for batch in train_loader:
        lr = tester.step()
        loss = ...
        tester.record(loss)
        optimizer.step()
        if tester.should_stop():
            break
    result = tester.suggest()

    # 方式 2: 一键函数（需要 trainer 实例）
    result = run_lr_range_test(trainer, dataloader, steps=500)
    print(f"Suggested head LR: {result['optimal_lr_head']}")
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LRRangeTest:
    """LR Range Test 执行器。

    每步指数增加学习率，记录 loss 变化，最终给出最优 LR 建议。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        steps: int = 500,
        smooth_window: int = 5,
    ):
        if len(optimizer.param_groups) == 0:
            raise ValueError("Optimizer has no param groups")

        self.optimizer = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = steps
        self.smooth_window = smooth_window

        # 计算指数增长因子: lr_i = start_lr * factor^i
        self.factor = (end_lr / start_lr) ** (1.0 / max(1, steps - 1))

        # 记录
        self.history: list[dict[str, float]] = []
        self._current_step = 0
        self._min_loss = float("inf")
        self._min_loss_step = 0
        self._best_lr: float | None = None

    def step(self) -> float:
        """将当前 LR 设为第 i 步的值并步进计数器。

        Returns:
            当前步骤的 LR（用于日志）。
        """
        if self._current_step >= self.total_steps:
            return self.current_lr()

        lr = self.start_lr * (self.factor ** self._current_step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._current_step += 1
        return lr

    def current_lr(self) -> float:
        """返回当前 LR 值（不步进）。"""
        return self.optimizer.param_groups[0]["lr"]

    def record(self, loss: float):
        """记录当前 (lr, loss) 对；（可选）应在 optimizer.step() 之后调用。"""
        lr = self.current_lr()
        entry = {"step": self._current_step, "lr": lr, "loss": loss}
        self.history.append(entry)

        if loss < self._min_loss:
            self._min_loss = loss
            self._min_loss_step = self._current_step

    def should_stop(self) -> bool:
        """判断是否应提前停止（loss 爆炸或超过最低 loss 的 4 倍）。"""
        if len(self.history) < 10:
            return False
        if self._current_step >= self.total_steps:
            return True
        recent = self.history[-1]["loss"]
        if math.isnan(recent) or math.isinf(recent):
            return True
        if self._min_loss > 0 and recent > 4.0 * self._min_loss:
            return True
        return False

    def smoothed_losses(self) -> list[float]:
        """返回平滑后的 loss 序列（简单移动平均）。"""
        losses = [h["loss"] for h in self.history]
        if len(losses) <= self.smooth_window:
            return losses
        smoothed = []
        w = self.smooth_window
        for i in range(len(losses)):
            start = max(0, i - w + 1)
            smoothed.append(sum(losses[start:i + 1]) / (i - start + 1))
        return smoothed

    def suggest(self, safety_factor: float = 10.0) -> dict[str, Any]:
        """分析 (lr, loss) 历史并给出 LR 建议。

        Args:
            safety_factor: 安全除数。suggested_lr = lr_at_min_loss / safety_factor
                           Leslie Smith 建议 10（取 loss 最低点 LR 的 1/10）。
                           默认 10.0（保守）。

        Returns:
            dict: {
                "optimal_lr": 推荐的基础 LR,
                "lr_at_min_loss": loss 最低点 LR,
                "lr_at_steepest": 下降最快点 LR,
                "min_loss": 最低 loss 值,
                "steps_tested": 测试步数,
                "history": [(lr, loss), ...]
            }
        """
        if len(self.history) < 5:
            return {
                "optimal_lr": None,
                "error": f"Insufficient data: {len(self.history)} points",
                "steps_tested": len(self.history),
                "history": [(h["lr"], h["loss"]) for h in self.history],
            }

        losses = self.smoothed_losses()
        lrs = [h["lr"] for h in self.history]

        # 找 loss 最低点对应的 LR
        min_idx = losses.index(min(losses))
        lr_at_min = lrs[min_idx]
        min_loss = losses[min_idx]

        # 找 loss 下降最快点（梯度绝对值最大）
        if len(losses) >= 3:
            slopes = [(losses[i] - losses[i - 1]) / max(lrs[i] - lrs[i - 1], 1e-15)
                      for i in range(1, len(losses))]
            steepest_idx = slopes.index(min(slopes))  # 最负斜率
            lr_at_steepest = lrs[steepest_idx]
        else:
            lr_at_steepest = lr_at_min / 10.0  # fallback

        # Leslie Smith 标准方法: optimal_lr = lr_at_min_loss / 10
        optimal_lr = lr_at_min / safety_factor

        # 同时考虑 steepest descent 点（取两者中较大的，但不超过 lr_at_min/3）
        alt_lr = abs(lr_at_steepest) * 10.0  # steepest 通常在极小 LR，放大 10x
        if alt_lr > optimal_lr and lr_at_steepest > 1e-6:
            optimal_lr = min(alt_lr, lr_at_min / 3.0)

        # 确保推荐值在合理范围内
        optimal_lr = max(1e-6, min(optimal_lr, lr_at_min * 0.5))

        return {
            "optimal_lr": optimal_lr,
            "lr_at_min_loss": lr_at_min,
            "lr_at_steepest_descent": lr_at_steepest,
            "min_loss": min_loss,
            "safety_factor": safety_factor,
            "steps_tested": len(self.history),
            "history": [(h["lr"], h["loss"]) for h in self.history],
            "smoothed_history": [(lrs[i], losses[i]) for i in range(len(lrs))],
        }


def run_lr_range_test(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int = 500,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    safety_factor: float = 3.0,
) -> dict[str, Any]:
    """一键运行 LR Range Test。

    自动将模型设为训练模式，执行 test，然后恢复原始 LR。

    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        steps: 测试步数（默认 500）
        start_lr: 起始 LR（默认 1e-7）
        end_lr: 终止 LR（默认 1.0）
        safety_factor: 推荐 LR 的安全除数（默认 3.0）

    Returns:
        LRRangeTest.suggest() 的结果字典。
    """
    model.train()
    original_lrs = [g["lr"] for g in optimizer.param_groups]

    tester = LRRangeTest(
        optimizer=optimizer,
        start_lr=start_lr,
        end_lr=end_lr,
        steps=steps,
    )

    data_iter = iter(train_loader)
    t_start = time.time()

    for _ in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        gt_boxes = batch["boxes"]
        gt_labels = batch["labels"]

        lr = tester.step()

        cls_pred, loc_pred = model(images)
        default_boxes = model.head.all_default_boxes(device)
        cls_loss, loc_loss = criterion(cls_pred, loc_pred, default_boxes, gt_boxes, gt_labels)
        loss = cls_loss + loc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tester.record(loss.item())

        if tester.should_stop():
            break

    elapsed = time.time() - t_start

    # 恢复原始 LR
    for group, orig_lr in zip(optimizer.param_groups, original_lrs):
        group["lr"] = orig_lr

    result = tester.suggest(safety_factor=safety_factor)
    result["elapsed_seconds"] = elapsed
    result["lr_per_second"] = tester._current_step / max(elapsed, 0.1)

    return result


def suggest_lr(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int = 500,
) -> float:
    """简化接口：返回推荐的基础 LR 值。

    Returns:
        float: 推荐的基础学习率（用于 head），或 None（测试失败时）。
    """
    result = run_lr_range_test(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        steps=steps,
    )
    return result.get("optimal_lr")
