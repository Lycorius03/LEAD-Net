"""自适应速度控制器 v4 — 五级精细面积-距离映射 + EMA 平滑。

原理:
    面积 ∝ 1/距离²（针孔相机模型）

    v4 五级映射（用户定义）:
        面积占比 < 5%   → 高速 (scale=1.0)
        面积 5-15%      → 中速 (scale=0.7)
        面积 15-20%     → 慢速 (scale=0.4)
        面积 20-30%     → 极慢 (scale=0.2)
        面积 > 30%      → 停止 (scale=0.0)

参考:
    - CBF-based Visual Servoing (Robotics and Autonomous Systems, Dec 2024)
    - ViKi-HyCo: IBVS + bbox velocity computation (arXiv:2311.07268)
    - 经典 IBVS (Image-Based Visual Servoing)

UART 协议完全不变。仅在面积值上做缩放。
"""

from __future__ import annotations

from .types import MotionConfig


class SpeedController:
    """自适应速度控制器（v4 五级映射）。

    用法::

        ctrl = SpeedController(config)
        new_a, new_cx, new_cy, state = ctrl.apply(area, cx, cy, image_area=320*240)
        # new_a 替代原始 a 写入 DecisionResult
    """

    def __init__(self, config: MotionConfig | None = None):
        self._cfg = config or MotionConfig()
        self._ema_area: float | None = None

    def apply(
        self, area: float, cx: float, cy: float,
        image_area: float = 76800.0,  # 320×240
    ) -> tuple[float, float, float, str]:
        """计算速度控制修正（v4 五级映射）。

        Args:
            area: 当前 bbox 面积 (w × h, px²)
            cx, cy: 当前目标中心坐标
            image_area: 图像总面积 (px²)，默认 320×240=76800

        Returns:
            (new_area, new_cx, new_cy, state_hint):
                new_area   — 修正后的面积（缩放后）
                new_cx     — 修正后的 cx
                new_cy     — 修正后的 cy
                state_hint — "tracking"|"approaching"|"slow"|"very_slow"|"braking"
        """
        cfg = self._cfg

        # 1. EMA 平滑
        if self._ema_area is None:
            self._ema_area = area
        else:
            self._ema_area = (
                cfg.speed_ema_beta * area
                + (1.0 - cfg.speed_ema_beta) * self._ema_area
            )

        smooth_area = self._ema_area
        area_ratio = smooth_area / max(image_area, 1.0)

        # 2. 五级映射
        if area_ratio > cfg.speed_area_very_slow:           # >30% → 停止
            new_a = area * cfg.speed_scale_stop
            new_cx, new_cy = cx, cy + cfg.speed_cy_pushback
            state_hint = "braking"
        elif area_ratio > cfg.speed_area_slow:              # 20-30% → 极慢
            new_a = area * cfg.speed_scale_very_slow
            new_cx, new_cy = cx, cy
            state_hint = "very_slow"
        elif area_ratio > cfg.speed_area_medium:            # 15-20% → 慢速
            new_a = area * cfg.speed_scale_slow
            new_cx, new_cy = cx, cy
            state_hint = "slow"
        elif area_ratio > cfg.speed_area_fast:              # 5-15% → 中速
            new_a = area * cfg.speed_scale_medium
            new_cx, new_cy = cx, cy
            state_hint = "approaching"
        else:                                                # <5% → 高速
            new_a = area * cfg.speed_scale_fast
            new_cx, new_cy = cx, cy
            state_hint = "tracking"

        # 3. 裁剪坐标
        new_cx = max(0.0, min(320.0, new_cx))
        new_cy = max(0.0, min(240.0, new_cy))

        return (new_a, new_cx, new_cy, state_hint)

    def get_area_ratio(self) -> float | None:
        """返回当前 EMA 面积占比（用于日志/FSM）。"""
        if self._ema_area is None:
            return None
        return self._ema_area / 76800.0

    def reset(self) -> None:
        """重置 EMA 累积器。"""
        self._ema_area = None
