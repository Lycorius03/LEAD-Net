"""路径记忆 + 轨迹预测器 — 混合预测 + 螺旋搜索。

混合预测策略 (PLOS ONE 2024 验证):
    - 短遮挡 (≤3 帧): 最小二乘线性外推（小样本下比 Kalman 更准）
    - 长遮挡 (>3 帧): Kalman 纯预测（利用速度状态）

路径记忆:
    - 环形缓冲记录最近 N 个已知位置
    - 丢失后逐步扩大螺旋搜索半径
    - 黄金角 137.5° 保证搜索方向均匀分布

参考:
    - PLOS ONE 2024: 混合最小二乘+Kalman 预测, ID Switch 减少 37%
    - IET Cyber-Systems 2024: 路径记忆库自主搜索
    - Kalman 滤波器 (已有 lead_net/tracking/kalman_filter.py)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from .types import MotionConfig


class TrajectoryPredictor:
    """混合轨迹预测器。

    用法::

        pred = TrajectoryPredictor(config)
        cx, cy = pred.predict(history, frames_lost)
    """

    def __init__(self, config: MotionConfig | None = None):
        self._cfg = config or MotionConfig()

    def predict(
        self,
        history: list[tuple[float, float, float, float]],
        frames_lost: int,
        vx: float = 0.0,
        vy: float = 0.0,
    ) -> tuple[float, float]:
        """预测 frames_lost 帧后的目标位置。

        Args:
            history: [(cx, cy, w, h), ...] 最近 N 帧历史（最新在末尾）
            frames_lost: 已丢失帧数
            vx, vy: Kalman 速度估计（用于长遮挡）

        Returns:
            (predicted_cx, predicted_cy)
        """
        if not history:
            return (160.0, 160.0)  # 默认图像中心

        if frames_lost <= self._cfg.predict_short_occlusion:
            return self._least_squares_predict(history, frames_lost)
        else:
            return self._kalman_extrapolate(history, frames_lost, vx, vy)

    def _least_squares_predict(
        self,
        history: list[tuple[float, float, float, float]],
        steps: int,
    ) -> tuple[float, float]:
        """最小二乘线性外推。

        x(t) = a_x × t + b_x, 外推 t = len(history) + steps - 1
        """
        n = len(history)
        if n < 2:
            last = history[-1]
            return (last[0], last[1])

        # 仅用最近 min(n, 5) 个点
        pts = history[-min(n, 5):]
        m = len(pts)
        # t 从 0 到 m-1
        t_mean = (m - 1) / 2.0
        cx_list = [p[0] for p in pts]
        cy_list = [p[1] for p in pts]
        cx_mean = sum(cx_list) / m
        cy_mean = sum(cy_list) / m

        # 斜率 a = Σ(t_i - t̄)(x_i - x̄) / Σ(t_i - t̄)²
        num_x, num_y, den = 0.0, 0.0, 0.0
        for i in range(m):
            dt = i - t_mean
            num_x += dt * (cx_list[i] - cx_mean)
            num_y += dt * (cy_list[i] - cy_mean)
            den += dt * dt

        if abs(den) < 1e-8:
            return (cx_list[-1], cy_list[-1])

        a_x = num_x / den
        a_y = num_y / den
        b_x = cx_mean - a_x * t_mean
        b_y = cy_mean - a_y * t_mean

        # 外推 t = m - 1 + steps
        t_pred = (m - 1) + steps
        pred_cx = a_x * t_pred + b_x
        pred_cy = a_y * t_pred + b_y

        return (pred_cx, pred_cy)

    @staticmethod
    def _kalman_extrapolate(
        history: list[tuple[float, float, float, float]],
        steps: int,
        vx: float,
        vy: float,
    ) -> tuple[float, float]:
        """Kalman 匀速模型外推。

        cx[k] = cx[0] + vx × k
        """
        last = history[-1]
        # 限制外推步数，防止发散
        k = min(steps, 10)
        return (last[0] + vx * k, last[1] + vy * k)


class PathMemory:
    """路径记忆 — 环形缓冲 + 螺旋搜索模式。

    用法::

        mem = PathMemory(config)
        mem.record(cx, cy)                   # 正常追踪时记录
        cx, cy = mem.get_search_pos(step)    # 丢失后获取搜索位置
    """

    def __init__(self, config: MotionConfig | None = None):
        self._cfg = config or MotionConfig()
        self._buffer: deque[tuple[float, float]] = deque(
            maxlen=self._cfg.memory_max_len,
        )

    def record(self, cx: float, cy: float) -> None:
        """记录一个已知位置。"""
        self._buffer.append((float(cx), float(cy)))

    @property
    def last_position(self) -> tuple[float, float] | None:
        """返回最后已知位置。"""
        return self._buffer[-1] if self._buffer else None

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def get_search_pos(self, step: int) -> tuple[float, float]:
        """生成螺旋搜索位置。

        黄金角 137.5° → 方向均匀分布，避免重复搜索同方向。
        半径逐步扩大: r = start + step × step_size, 上限 max。

        Args:
            step: 搜索步数（从 0 开始）

        Returns:
            (search_cx, search_cy): 建议搜索坐标
        """
        cfg = self._cfg
        if self.is_empty:
            return (160.0, 160.0)

        last = self._buffer[-1]
        radius = min(
            cfg.memory_search_radius_start + step * cfg.memory_search_radius_step,
            cfg.memory_search_radius_max,
        )
        angle_deg = (step * cfg.memory_golden_angle) % 360.0
        angle_rad = math.radians(angle_deg)

        cx = last[0] + radius * math.cos(angle_rad)
        cy = last[1] + radius * math.sin(angle_rad)

        # 裁剪到有效范围
        cx = max(0.0, min(320.0, cx))
        cy = max(0.0, min(240.0, cy))
        return (cx, cy)

    def get_path_direction(self) -> tuple[float, float]:
        """返回最近运动方向（用于预测）。"""
        if len(self._buffer) < 2:
            return (0.0, 0.0)
        p1 = self._buffer[-2]
        p2 = self._buffer[-1]
        return (p2[0] - p1[0], p2[1] - p1[1])

    def reset(self) -> None:
        """清空路径记忆。"""
        self._buffer.clear()
