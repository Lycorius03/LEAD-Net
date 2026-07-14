"""ROI 空间过滤器 —— 三层决策流程第二层。

仅保留 bbox 中心落在避障关注区（ROI）内的目标。
ROI 定义为画面中央水平区域 + 下半垂直区域。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ROIParams:
    """ROI 区域参数（归一化比例，相对画面尺寸）。"""

    h_start: float = 0.175   # 水平起点（左边界，占画面宽度比例）
    h_end: float = 0.825     # 水平终点（右边界）
    v_start: float = 0.40    # 垂直起点（上边界）
    v_end: float = 1.0       # 垂直终点（下边界，1.0=画面底部）

    @classmethod
    def from_cfg(cls, cfg: dict) -> "ROIParams":
        d = cfg.get("decision", {})
        hr = d.get("roi_horizontal_range", [0.175, 0.825])
        vr = d.get("roi_vertical_range", [0.40, 1.0])
        return cls(h_start=hr[0], h_end=hr[1], v_start=vr[0], v_end=vr[1])

    def contains(self, cx: float, cy: float, img_w: int, img_h: int) -> bool:
        """检查 (cx, cy) 像素坐标是否落在 ROI 内。"""
        x_ok = self.h_start * img_w <= cx <= self.h_end * img_w
        y_ok = self.v_start * img_h <= cy <= self.v_end * img_h
        return x_ok and y_ok


class ROIFilter:
    """ROI 空间过滤器。

    用法::

        roi = ROIFilter(params)
        filtered = roi.apply(detections, image_width, image_height)
    """

    def __init__(self, params: ROIParams | None = None):
        self.params = params or ROIParams()

    def apply(
        self,
        detections: list[dict[str, Any]],
        image_width: int = 320,
        image_height: int = 320,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """过滤检测结果。

        Args:
            detections: [{"bbox": [cx,cy,w,h], "score": float, ...}, ...]
            image_width: 图像宽度（像素）
            image_height: 图像高度（像素）

        Returns:
            (in_roi, out_of_roi): 落在 ROI 内/外的检测列表
        """
        inside, outside = [], []
        for det in detections:
            bbox = det["bbox"]
            cx, cy = bbox[0], bbox[1]
            if self.params.contains(cx, cy, image_width, image_height):
                inside.append(det)
            else:
                outside.append(det)
        return inside, outside
