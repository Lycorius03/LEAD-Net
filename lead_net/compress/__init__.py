"""模型压缩模块 — 结构化通道剪枝。

论文: Network Slimming (ICCV 2017), arXiv:2507.16155 (2025, STM32H7验证)
"""
from lead_net.compress.channel_pruner import ChannelPruner, slim_bn_gamma

__all__ = ["ChannelPruner", "slim_bn_gamma"]
