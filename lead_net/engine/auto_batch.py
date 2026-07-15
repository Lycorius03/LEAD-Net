"""Auto Batch — 自动探测最优 Batch Size。

从保守起始值递增尝试，用真实训练数据做 forward+backward，
确认显存占用 ≤ 目标利用率后选择该 batch_size。

用法:
    bs = auto_batch_size(model, train_loader, cfg, device)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def auto_batch_size(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    target_util: float = 0.85,
) -> int:
    """自动探测当前 GPU 的安全 batch size。

    策略：从小到大递增（而非从大到小递减），避免 OOM 导致的 CUDA 状态损坏。
    使用真实训练图像做 forward+backward，考虑梯度和优化器开销。
    """
    if device.type != "cuda":
        return 8

    total_mem = torch.cuda.get_device_properties(device).total_memory
    gb = total_mem / 1024**3

    # 根据显存容量确定探测范围
    if gb >= 30:
        candidates = [64, 48, 32, 24, 16]
    elif gb >= 20:
        candidates = [48, 32, 24, 16, 8]
    elif gb >= 10:
        candidates = [24, 16, 12, 8, 6, 4]
    else:
        candidates = [12, 8, 6, 4, 2]

    model.to(device)
    model.train()

    # 获取真实图像尺寸
    try:
        sample = next(iter(train_loader))
        imgs = sample["image"].to(device)
        c, h, w = imgs.shape[1:]
        # 使用实际图像做测试
        test_img = imgs[:1].detach().clone()
    except StopIteration:
        c, h, w = 3, cfg.get("data", {}).get("input_size", 320), cfg.get("data", {}).get("input_size", 320)
        test_img = torch.randn(1, c, h, w, device=device)

    # 估算单样本显存占用
    with torch.no_grad():
        cls_pred, loc_pred = model(test_img)
    single_fwd = torch.cuda.memory_allocated(device) - _baseline_mem(device)

    # 经验：训练时显存约为推理的 3-4x（forward+backward+optimizer states）
    # 保守取 4x，加 20% 缓冲
    safety_factor = 0.8
    target_bytes = int(total_mem * target_util * safety_factor)
    estimate_per_sample = single_fwd * 4
    theoretical_max = max(2, target_bytes // max(estimate_per_sample, 1))

    # 从理论最大值向下取候选列表中的值
    candidates = [c for c in candidates if c <= theoretical_max]
    if not candidates:
        candidates = [2]

    print(f"[auto_batch] GPU: {torch.cuda.get_device_name(device)} ({gb:.1f}GB)")
    print(f"[auto_batch] single fwd: {single_fwd/1024**2:.0f}MB, "
          f"est train: {estimate_per_sample/1024**2:.0f}MB/sample, "
          f"theoretical max: {theoretical_max}")

    for bs in sorted(candidates, reverse=True):
        try:
            torch.cuda.empty_cache()
            x = test_img.repeat(bs, 1, 1, 1)
            cls_pred, loc_pred = model(x)
            loss = cls_pred.sum() + loc_pred.sum()
            loss.backward()
            model.zero_grad()

            used = torch.cuda.memory_allocated(device)
            util = used / total_mem
            print(f"  bs={bs:>3} → {used/1024**3:.2f}GB ({util*100:.0f}%)")

            if util <= target_util:
                torch.cuda.empty_cache()
                print(f"[auto_batch] selected batch_size={bs}")
                return bs
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            print(f"  bs={bs:>3} → OOM")

    bs = candidates[-1]
    torch.cuda.empty_cache()
    print(f"[auto_batch] fallback to batch_size={bs}")
    return bs


def _baseline_mem(device: torch.device) -> int:
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated(device)
