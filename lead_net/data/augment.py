"""数据增强模块 — CutOut + RandomErase。

论文:
    - CutOut: DeVries & Taylor, "Improved Regularization of CNNs with Cutout", arXiv:2017
    - Random Erasing: Zhong et al., "Random Erasing Data Augmentation", AAAI 2020

用途:
    - 模拟遮挡，提升模型在部分遮挡下的检测鲁棒性
    - 对追踪重拾场景特别有价值
"""

from __future__ import annotations

import random
import torch


def cutout(image: torch.Tensor, n_holes: int = 1, hole_size: int = 32,
           fill_value: float = 0.0) -> torch.Tensor:
    """CutOut 增强 — 在图像上随机放置灰色方块。

    Args:
        image: [C, H, W] tensor (已在 [0,1] 或归一化后)
        n_holes: 遮挡方块数
        hole_size: 方块边长 (px)
        fill_value: 填充值（0=黑色，ImageNet 归一化后为 -mean/std）

    Returns:
        增强后的图像
    """
    _, h, w = image.shape
    result = image.clone()

    for _ in range(n_holes):
        x = random.randint(0, max(1, w - hole_size))
        y = random.randint(0, max(1, h - hole_size))
        # 实际 hole 尺寸可随机变化
        actual_h = min(random.randint(hole_size // 2, hole_size), h - y)
        actual_w = min(random.randint(hole_size // 2, hole_size), w - x)
        result[:, y:y + actual_h, x:x + actual_w] = fill_value

    return result


def random_erase(image: torch.Tensor, p: float = 0.5,
                 scale: tuple[float, float] = (0.02, 0.33),
                 ratio: tuple[float, float] = (0.3, 3.3),
                 fill_value: float = 0.0) -> torch.Tensor:
    """Random Erasing 增强（AAAI 2020）。

    Args:
        image: [C, H, W] tensor
        p: 执行概率
        scale: 擦除区域面积比例范围
        ratio: 擦除区域宽高比范围
        fill_value: 填充值

    Returns:
        增强后的图像
    """
    if random.random() > p:
        return image

    _, h, w = image.shape
    area = h * w

    for _ in range(10):  # 最多尝试 10 次
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        erase_h = int(round((target_area * aspect_ratio) ** 0.5))
        erase_w = int(round((target_area / aspect_ratio) ** 0.5))

        if erase_w < w and erase_h < h:
            x = random.randint(0, w - erase_w)
            y = random.randint(0, h - erase_h)
            image[:, y:y + erase_h, x:x + erase_w] = fill_value
            break

    return image


class CutOutAugment:
    """CutOut 增强包装器（与其他 transform 兼容）。

    Args:
        n_holes: 遮挡方块数
        hole_size: 方块最大边长
        p: 执行概率
        fill: 填充值
    """

    def __init__(self, n_holes: int = 1, hole_size: int = 40,
                 p: float = 0.5, fill: float = 0.0):
        self.n_holes = n_holes
        self.hole_size = hole_size
        self.p = p
        self.fill = fill

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return image
        return cutout(image, self.n_holes, self.hole_size, self.fill)


class RandomEraseAugment:
    """Random Erasing 增强包装器。

    Args:
        p: 执行概率
        scale: 面积比例范围
    """

    def __init__(self, p: float = 0.5, scale: tuple[float, float] = (0.02, 0.2)):
        self.p = p
        self.scale = scale

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return random_erase(image, self.p, self.scale)
