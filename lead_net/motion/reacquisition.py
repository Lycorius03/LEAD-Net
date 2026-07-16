"""目标重拾搜索模块 — 螺旋搜索 + ROI 扩展 + 路径记忆。

论文依据：
    - JIRS Vol.110 (2024): 螺旋搜索用于机器人重连
    - IET Cyber-Systems (2024): 路径记忆 + 特征记忆库自主搜索
    - 阿基米德螺旋: r(θ) = a + b·θ，黄金角 137.5°

核心思路：
    目标因障碍遮挡而丢失后，不立即回到全局搜索，
    而是从最后已知位置出发，逐步扩大搜索范围，尝试重拾。

用法::

    reacq = ReacquisitionEngine(config)
    reacq.start(last_cx, last_cy, last_w, last_h)
    for frame in range(max_frames):
        cx, cy, expanded_roi = reacq.get_search_params(frame)
        # 在 expanded_roi 区域内检测
        if target_detected:
            reacq.on_reacquired()
            break
    if reacq.is_timed_out():
        # 回退到全局 SEARCHING
"""

from __future__ import annotations

import math
from .types import MotionConfig


class ReacquisitionEngine:
    """目标重拾搜索引擎。

    阿基米德螺旋搜索 + ROI 逐步扩展 + 自适应置信度阈值。

    Args:
        config: MotionConfig 或 None（使用默认值）。
    """

    def __init__(self, config: MotionConfig | None = None):
        self._cfg = config or MotionConfig()
        self._last_cx: float = 160.0
        self._last_cy: float = 120.0
        self._last_w: float = 50.0
        self._last_h: float = 50.0
        self._frame_count: int = 0
        self._active: bool = False

    # ---- 公共 API ----

    def start(self, last_cx: float, last_cy: float,
              last_w: float = 50.0, last_h: float = 50.0) -> None:
        """开始重拾搜索，记录目标最后已知状态。

        Args:
            last_cx, last_cy: 目标最后已知中心坐标 (px)
            last_w, last_h: 目标最后已知尺寸 (px)
        """
        self._last_cx = float(last_cx)
        self._last_cy = float(last_cy)
        self._last_w = max(float(last_w), 10.0)
        self._last_h = max(float(last_h), 10.0)
        self._frame_count = 0
        self._active = True

    def get_search_params(self, step: int | None = None) -> tuple[float, float, dict]:
        """获取当前帧的搜索参数。

        Args:
            step: 搜索步数，None 则使用内部计数器。

        Returns:
            (search_cx, search_cy, roi_info):
                search_cx, search_cy — 建议搜索中心坐标
                roi_info — dict with keys:
                    - expand_factor: ROI 扩展倍数
                    - confidence_threshold: 自适应置信度阈值
                    - search_radius: 当前搜索半径 (px)
        """
        cfg = self._cfg
        s = step if step is not None else self._frame_count
        self._frame_count = max(self._frame_count, s + 1)

        # 1. 螺旋搜索位置（阿基米德螺旋: r = a + b * step）
        radius = cfg.reacquire_spiral_a + cfg.reacquire_spiral_b * s
        angle_deg = (s * cfg.memory_golden_angle) % 360.0
        angle_rad = math.radians(angle_deg)

        search_cx = self._last_cx + radius * math.cos(angle_rad)
        search_cy = self._last_cy + radius * math.sin(angle_rad)

        # 裁剪到有效范围 (320 × 240)
        search_cx = max(0.0, min(320.0, search_cx))
        search_cy = max(0.0, min(240.0, search_cy))

        # 2. ROI 逐步扩展
        expand_factor = min(
            1.0 + cfg.reacquire_roi_expand_rate * s,
            cfg.reacquire_roi_max_expand,
        )

        # 3. 自适应置信度阈值（越往后越宽松）
        confidence_scale = max(cfg.reacquire_confidence_boost, 1.0 - 0.02 * s)
        base_threshold = 0.3  # 默认置信度阈值
        adaptive_threshold = base_threshold * confidence_scale

        roi_info = {
            "expand_factor": expand_factor,
            "confidence_threshold": adaptive_threshold,
            "search_radius": radius,
            "roi_cx": search_cx,
            "roi_cy": search_cy,
            "roi_w": self._last_w * expand_factor,
            "roi_h": self._last_h * expand_factor,
        }

        return (search_cx, search_cy, roi_info)

    @property
    def last_position(self) -> tuple[float, float]:
        """目标最后已知位置。"""
        return (self._last_cx, self._last_cy)

    @property
    def last_size(self) -> tuple[float, float]:
        """目标最后已知尺寸。"""
        return (self._last_w, self._last_h)

    @property
    def frame_count(self) -> int:
        """当前重拾阶段已用帧数。"""
        return self._frame_count

    @property
    def is_active(self) -> bool:
        return self._active

    def is_timed_out(self) -> bool:
        """检查是否已超时。"""
        return self._frame_count >= self._cfg.reacquire_max_frames

    def on_reacquired(self) -> None:
        """目标重拾成功时的回调。"""
        self._active = False

    def get_expanded_roi_bbox(self, step: int | None = None) -> tuple[float, float, float, float]:
        """获取扩展后的 ROI 边界框。

        Returns:
            (roi_cx, roi_cy, roi_w, roi_h): 扩展后的搜索区域
        """
        _, _, info = self.get_search_params(step)
        return (info["roi_cx"], info["roi_cy"], info["roi_w"], info["roi_h"])

    def reset(self) -> None:
        """重置搜索引擎。"""
        self._last_cx = 160.0
        self._last_cy = 120.0
        self._last_w = 50.0
        self._last_h = 50.0
        self._frame_count = 0
        self._active = False


def is_point_in_roi(px: float, py: float,
                    roi_cx: float, roi_cy: float,
                    roi_w: float, roi_h: float) -> bool:
    """检查点是否在 ROI 区域内。

    Args:
        px, py: 待检查点坐标
        roi_cx, roi_cy: ROI 中心
        roi_w, roi_h: ROI 宽高

    Returns:
        True 如果点在 ROI 内。
    """
    half_w = roi_w / 2.0
    half_h = roi_h / 2.0
    return (abs(px - roi_cx) <= half_w and abs(py - roi_cy) <= half_h)


def compute_iou_with_roi(box: tuple[float, float, float, float],
                         roi: tuple[float, float, float, float]) -> float:
    """计算检测框与 ROI 区域的 IoU。

    Args:
        box: (cx, cy, w, h) 检测框
        roi: (cx, cy, w, h) ROI 区域

    Returns:
        IoU ∈ [0, 1]。
    """
    # 转为 xyxy
    def to_xyxy(cx, cy, w, h):
        return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)

    b = to_xyxy(*box)
    r = to_xyxy(*roi)

    # 交集
    ix1 = max(b[0], r[0])
    iy1 = max(b[1], r[1])
    ix2 = min(b[2], r[2])
    iy2 = min(b[3], r[3])

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_r = (r[2] - r[0]) * (r[3] - r[1])
    union = area_b + area_r - inter

    return inter / union if union > 0 else 0.0
