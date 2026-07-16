"""BN-γ 结构化通道剪枝（Network Slimming）。

论文: Liu et al., "Learning Efficient CNNs via Network Slimming", ICCV 2017

原理:
    1. 对 BN 层的 γ 参数施加 L1 正则化（稀疏训练）
    2. 训练后，γ≈0 的通道被认为不重要
    3. 剪掉这些通道及其对应的卷积滤波器
    4. Fine-tune 恢复精度

STM32H7 验证: arXiv:2507.16155 (2025) — YOLOv5n 在 STM32H743 上成功部署

用法::

    # Step 1: 稀疏训练（添加 BN L1 正则化）
    pruner = ChannelPruner(model)
    pruner.enable_sparse_training(l1_lambda=1e-4)

    # Step 2: 剪枝
    pruner.prune(percent=0.3)  # 剪掉 30% 通道

    # Step 3: Fine-tune
    # ... 正常训练 ...
"""

from __future__ import annotations

import torch
import torch.nn as nn


def slim_bn_gamma(model: nn.Module, l1_lambda: float = 1e-4) -> torch.Tensor:
    """计算 BN-γ 的 L1 稀疏正则化项。

    Args:
        model: 模型
        l1_lambda: L1 正则化强度

    Returns:
        L1 loss (scalar)，需加到总 loss 中
    """
    l1_loss = torch.tensor(0.0)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            l1_loss = l1_loss + m.weight.abs().sum()
    return l1_loss * l1_lambda


class ChannelPruner:
    """BN-γ 通道剪枝器。

    Args:
        model: 待剪枝模型
        prune_ratio: 全局剪枝比例（0.0~1.0）
    """

    def __init__(self, model: nn.Module, prune_ratio: float = 0.3):
        self.model = model
        self.prune_ratio = prune_ratio
        self._sparse_enabled = False
        self._pruning_mask: dict[str, torch.Tensor] = {}

    def enable_sparse_training(self, l1_lambda: float = 1e-4) -> None:
        """启用 BN-γ 稀疏训练。"""
        self._sparse_enabled = True
        self.l1_lambda = l1_lambda

    def get_sparsity_loss(self) -> torch.Tensor:
        """获取当前模型的 BN-γ L1 稀疏损失。"""
        if not self._sparse_enabled:
            return torch.tensor(0.0)
        return slim_bn_gamma(self.model, self.l1_lambda)

    def prune(self, percent: float | None = None) -> dict[str, int]:
        """执行剪枝。

        Args:
            percent: 剪枝比例，None 则使用初始化值

        Returns:
            {"pruned_channels": N, "remaining_channels": M}
        """
        p = percent if percent is not None else self.prune_ratio

        # 收集所有 BN 层的 γ 值
        all_gamma = []
        bn_modules = []
        for name, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                all_gamma.extend(m.weight.data.abs().tolist())
                bn_modules.append((name, m))

        if not all_gamma:
            return {"pruned_channels": 0, "remaining_channels": 0}

        # 计算全局阈值
        threshold = _find_percentile_threshold(all_gamma, p)

        # 生成 mask
        total_pruned = 0
        total_channels = 0
        for name, m in bn_modules:
            mask = m.weight.data.abs() > threshold
            pruned = (~mask).sum().item()
            total_pruned += pruned
            total_channels += len(mask)
            self._pruning_mask[name] = mask
            # 将剪掉的 γ 置零
            m.weight.data[~mask] = 0.0

        return {"pruned_channels": total_pruned, "remaining_channels": total_channels - total_pruned}

    def get_mask(self) -> dict[str, torch.Tensor]:
        """获取当前剪枝 mask。"""
        return dict(self._pruning_mask)

    def estimate_compression_ratio(self) -> float:
        """估算剪枝后的参数压缩比。"""
        total = 0
        pruned = 0
        for name, mask in self._pruning_mask.items():
            total += len(mask)
            pruned += (~mask).sum().item()
        return pruned / total if total > 0 else 0.0

    @staticmethod
    def log_sparsity(model: nn.Module) -> str:
        """统计当前 BN-γ 稀疏度，返回可读摘要。"""
        total_channels = 0
        near_zero = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                g = m.weight.data.abs()
                total_channels += len(g)
                near_zero += (g < 1e-4).sum().item()
        pct = 100.0 * near_zero / total_channels if total_channels > 0 else 0.0
        return f"BN sparsity: {near_zero}/{total_channels} = {pct:.1f}%"


def _find_percentile_threshold(values: list[float], percent: float) -> float:
    """找到第 percent 分位数值。"""
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * percent)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx] if sorted_vals else 0.0
