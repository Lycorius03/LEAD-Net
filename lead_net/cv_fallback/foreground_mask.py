"""前景掩膜生成器 —— 组合地面分割 + blob 检测，输出障碍物候选框。

供 DecisionEngine 的融合前置使用（DL 检测 + CV 前景做 IoU 重叠验证）。

管道：
    image → GroundSegmenter → ground_mask → 取反 → foreground_mask
          → BlobDetector → 障碍物候选框 list[{"bbox": [cx,cy,w,h], "score": float}]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ground_segmenter import GroundSegmenter, GroundSegmenterParams
from .blob_detector import BlobDetector, BlobParams


@dataclass
class CvFallbackParams:
    """传统 CV 兜底模块总参数。"""

    ground: GroundSegmenterParams = None  # type: ignore
    blob: BlobParams = None  # type: ignore

    def __post_init__(self):
        if self.ground is None:
            self.ground = GroundSegmenterParams()
        if self.blob is None:
            self.blob = BlobParams()

    @classmethod
    def from_cfg(cls, cfg: dict) -> "CvFallbackParams":
        return cls(
            ground=GroundSegmenterParams.from_cfg(cfg),
            blob=BlobParams.from_cfg(cfg),
        )


class CvFallback:
    """传统 CV 前景检测管道。

    用法::

        cv = CvFallback(params)
        regions = cv.process(image)  # np.uint8 (H,W,3) → list[{"bbox":, "score":}]
    """

    def __init__(self, params: CvFallbackParams | None = None):
        self.params = params or CvFallbackParams()
        self._segmenter = GroundSegmenter(self.params.ground)
        self._detector = BlobDetector(self.params.blob)
        self._last_ground_mask: np.ndarray | None = None
        self._last_foreground_mask: np.ndarray | None = None

    def process(self, image: np.ndarray) -> list[dict[str, Any]]:
        """处理一帧图像，返回障碍物候选框列表。

        Args:
            image: (H, W, 3) np.uint8 RGB

        Returns:
            list[{"bbox": [cx,cy,w,h], "score": float}]
        """
        # 1. 地面分割
        ground_mask = self._segmenter.segment(image)
        self._last_ground_mask = ground_mask

        # 2. 取反得前景（潜在障碍物）
        foreground_mask = ~ground_mask
        self._last_foreground_mask = foreground_mask

        # 3. blob 检测
        blobs = self._detector.detect(foreground_mask)

        # 4. 为每个 blob 赋予置信度（基于密度和相对面积）
        h, w = image.shape[:2]
        total_pixels = h * w
        regions = []
        for blob in blobs:
            area_norm = blob["area"] / total_pixels
            # 置信度 = 密度 × sqrt(面积归一化)（大且密集的 blob 更可信）
            confidence = blob["density"] * min(1.0, area_norm * 10)
            regions.append({
                "bbox": blob["bbox"],
                "score": round(float(confidence), 4),
                "area": blob["area"],
                "density": blob["density"],
                "source": "CV",
            })

        return regions

    @property
    def ground_mask(self) -> np.ndarray | None:
        return self._last_ground_mask

    @property
    def foreground_mask(self) -> np.ndarray | None:
        return self._last_foreground_mask

    @property
    def dominant_color(self) -> np.ndarray | None:
        return self._segmenter.dominant_color
