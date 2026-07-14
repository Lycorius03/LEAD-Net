"""地面分割器 —— 基于颜色聚类的可通行区域识别。

原理：
    - 采样画面下半部分的主导颜色作为"地面参考色"
    - 对全图做颜色距离阈值分割，接近地面色的=可通行，远离的=潜在障碍
    - 不依赖深度学习，计算量极小，适合 OpenMV 等边缘设备

参考：
    - R20 (IndoorObstacleDiscovery-RG, IJCV 2024): 地面检测免疫反光
    - R19 (UAV自监督系统, 2025): 地面平面+自适应尺度因子
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GroundSegmenterParams:
    """地面分割参数。"""

    sample_ratio: float = 0.3       # 从画面底部采样的比例（下30%区域）
    color_threshold: float = 40.0    # 颜色距离阈值（欧氏距离，0-255 RGB空间）
    min_ground_ratio: float = 0.1   # 地面像素最低占比（低于此值认为分割失败）

    @classmethod
    def from_cfg(cls, cfg: dict) -> "GroundSegmenterParams":
        d = cfg.get("cv_fallback", {}).get("ground", {})
        return cls(
            sample_ratio=d.get("sample_ratio", 0.3),
            color_threshold=d.get("color_threshold", 40.0),
            min_ground_ratio=d.get("min_ground_ratio", 0.1),
        )


class GroundSegmenter:
    """基于颜色聚类的地面分割器。

    用法::

        seg = GroundSegmenter(params)
        ground_mask = seg.segment(image)  # image: (H, W, 3) np.uint8
        # ground_mask: (H, W) bool, True=地面/可通行
    """

    def __init__(self, params: GroundSegmenterParams | None = None):
        self.params = params or GroundSegmenterParams()
        self._dominant_color: np.ndarray | None = None  # (3,) RGB

    def segment(self, image: np.ndarray) -> np.ndarray:
        """分割地面区域。

        Args:
            image: (H, W, 3) np.uint8 RGB 图像

        Returns:
            ground_mask: (H, W) bool, True=地面/可通行, False=潜在障碍
        """
        h, w = image.shape[:2]

        # 1. 采样底部区域计算地面主导色
        sample_start = int(h * (1 - self.params.sample_ratio))
        ground_sample = image[sample_start:, :, :].reshape(-1, 3).astype(np.float64)
        self._dominant_color = np.median(ground_sample, axis=0)

        # 2. 全图像素级颜色距离
        diff = image.astype(np.float64) - self._dominant_color.reshape(1, 1, 3)
        distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (H, W)

        ground_mask = distances <= self.params.color_threshold

        # 3. 形态学清理：去除孤立噪点（简单的 3x3 多数滤波）
        ground_mask = self._denoise(ground_mask)

        # 4. 检查分割质量
        ground_ratio = ground_mask.sum() / (h * w)
        if ground_ratio < self.params.min_ground_ratio:
            # 分割失败（如强光照、纹理复杂地面），返回全 False（全部视为障碍）
            return np.zeros((h, w), dtype=bool)

        return ground_mask

    @property
    def dominant_color(self) -> np.ndarray | None:
        """最近一次分割的地面主导色 (3,) RGB。"""
        return self._dominant_color

    @staticmethod
    def _denoise(mask: np.ndarray) -> np.ndarray:
        """3x3 多数滤波去噪（向量化实现，不需要 OpenCV）。"""
        h, w = mask.shape
        padded = np.pad(mask.astype(np.int32), 1, mode="edge")
        # 向量化：将 9 个偏移视图求和
        result = sum(
            padded[dy:dy+h, dx:dx+w]
            for dy in range(3) for dx in range(3)
        )
        return result >= 5  # 3x3 中 ≥5 个 True → True
