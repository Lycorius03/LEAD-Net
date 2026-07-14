"""Blob 检测器 —— 连通分量标记 + 边界框提取。

在前景掩膜上运行两遍连通分量算法，提取每个 blob 的边界框。
不依赖 OpenCV，纯 numpy 实现，适合 OpenMV 移植。

参考：
    - OpenMV find_blobs() API 设计理念
    - R20 (IJCV 2024): 前景区域检测与边界框回归
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BlobParams:
    """Blob 检测参数。"""

    min_area: int = 50           # 最小 blob 面积（像素），过滤噪点
    max_area: int = 50000        # 最大 blob 面积
    min_density: float = 0.3    # 最小填充率（面积/边界框面积）

    @classmethod
    def from_cfg(cls, cfg: dict) -> "BlobParams":
        d = cfg.get("cv_fallback", {}).get("blob", {})
        return cls(
            min_area=d.get("min_area", 50),
            max_area=d.get("max_area", 50000),
            min_density=d.get("min_density", 0.3),
        )


class BlobDetector:
    """连通分量 Blob 检测器。

    用法::

        detector = BlobDetector(params)
        regions = detector.detect(foreground_mask)
        # regions: [{"bbox": [cx,cy,w,h], "area": int, "density": float}, ...]
    """

    def __init__(self, params: BlobParams | None = None):
        self.params = params or BlobParams()

    def detect(self, mask: np.ndarray) -> list[dict[str, Any]]:
        """在二值掩膜上检测连通区域。

        Args:
            mask: (H, W) bool, True=前景/障碍物

        Returns:
            list[dict]: 每个 blob 含 "bbox" [cx,cy,w,h], "area", "density"
        """
        if not mask.any():
            return []

        # 两遍连通分量标记（简化版，4-邻接）
        labels = self._connected_components(mask)
        if labels is None:
            return []

        # 提取每个 blob 的统计信息
        regions = []
        for lbl in range(1, int(labels.max()) + 1):
            ys, xs = np.where(labels == lbl)
            area = len(ys)
            if area < self.params.min_area or area > self.params.max_area:
                continue

            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            density = area / max(w * h, 1)

            if density < self.params.min_density:
                continue

            regions.append({
                "bbox": [
                    float(x1 + w / 2),  # cx
                    float(y1 + h / 2),  # cy
                    float(w),           # w
                    float(h),           # h
                ],
                "area": area,
                "density": round(density, 3),
            })

        return regions

    @staticmethod
    def _connected_components(mask: np.ndarray) -> np.ndarray | None:
        """两遍连通分量标记（4-邻接，向量化实现）。

        Returns:
            labels: (H, W) int32, 0=背景, 1..N=各连通分量
        """
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        label_id = 1
        equivalences: dict[int, int] = {}

        # 第一遍：扫描标记
        for y in range(h):
            for x in range(w):
                if not mask[y, x]:
                    continue
                # 检查上、左邻居
                up = labels[y - 1, x] if y > 0 else 0
                left = labels[y, x - 1] if x > 0 else 0
                if up == 0 and left == 0:
                    labels[y, x] = label_id
                    label_id += 1
                elif up != 0 and left != 0:
                    labels[y, x] = min(up, left)
                    if up != left:
                        a, b = max(up, left), min(up, left)
                        equivalences[a] = b
                else:
                    labels[y, x] = up if up != 0 else left

        if label_id == 1:
            return None  # 无连通分量

        # 等价类合并
        for lbl in range(1, label_id):
            root = lbl
            while root in equivalences:
                root = equivalences[root]
            labels[labels == lbl] = root

        return labels
