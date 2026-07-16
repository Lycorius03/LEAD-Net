"""APF 避障器 — 改进人工势场法（切线力 + 局部最小值逃逸）。

参考:
    - RAPF (arXiv:2405.16659): 细菌点采样 + 高斯代价 APF
    - Adaptive APF (ScienceDirect 2026): 切线力 + 动态参数调节
    - Laplace APF (TechRxiv 2024): 68μs 中位执行时间

关键创新: 障碍物 = 非追踪目标的其他检测框。不需要深度传感器！

算法 (O(n), n = 障碍物数 ≤ 3):
    F_att  = k_att × (P_center - P_target)             ← 吸引力（拉向视野中心）
    F_rep  = Σ k_rep × (1/d_i - 1/d0) / d_i² × dir_i  ← 排斥力（远离障碍物）
    F_tan  = Σ k_tan × mag_i × tangent_i               ← 切线力（绕过障碍物）
    P_out  = P_target + gain × (F_att + F_rep + F_tan) ← 修正坐标
"""

from __future__ import annotations

import math
from typing import Any

from .types import MotionConfig


class ObstacleAvoider:
    """APF 避障器。

    用法::

        avoider = ObstacleAvoider(config)
        cx_new, cy_new = avoider.apply(cx, cy, w, h, obstacles)
    """

    def __init__(self, config: MotionConfig | None = None):
        self._cfg = config or MotionConfig()
        self._image_center = (160.0, 160.0)  # 320×320 model space 中心
        self._stuck_counter: int = 0
        self._stuck_direction: tuple[float, float] = (0.0, 0.0)

    def apply(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        obstacles: list[dict[str, Any]],
    ) -> tuple[float, float]:
        """对追踪目标应用 APF 避障偏置。

        Args:
            cx, cy: 追踪目标中心坐标 (model space)
            w, h: 追踪目标宽高
            obstacles: 障碍物列表 [{"cx":, "cy":, "w":, "h":}, ...]

        Returns:
            (cx_new, cy_new): 修正后的坐标
        """
        cfg = self._cfg

        # 1. 吸引力（从目标位置指向图像中心）
        att_x = cfg.apf_k_att * (self._image_center[0] - cx)
        att_y = cfg.apf_k_att * (self._image_center[1] - cy)

        # 2. 排斥力 + 切线力
        rep_x, rep_y = 0.0, 0.0
        tan_x, tan_y = 0.0, 0.0
        has_obstacle = False

        for obs in obstacles:
            ox = obs.get("cx", obs.get("x", 0.0))
            oy = obs.get("cy", obs.get("y", 0.0))
            ow = obs.get("w", 0.0)
            oh = obs.get("h", 0.0)

            # 障碍物面积过滤
            if ow * oh < cfg.obstacle_min_area:
                continue

            dx = cx - ox
            dy = cy - oy
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 1e-6:
                dist = 1e-6  # 防止除零

            if dist < cfg.apf_d0:
                has_obstacle = True
                # 排斥力: F = k × (1/d - 1/d0) / d²
                mag = cfg.apf_k_rep * (1.0 / dist - 1.0 / cfg.apf_d0) / (dist * dist)
                if mag > 0:
                    nx = dx / dist  # 从障碍物指向目标的单位向量
                    ny = dy / dist
                    rep_x += mag * nx
                    rep_y += mag * ny

                # 切线力: 垂直于排斥力方向，选择靠近目标中心的一侧
                # 左旋90°: (-ny, nx)，右旋90°: (ny, -nx)
                # 选点积大的一侧（与 center→target 方向一致）
                to_center_x = self._image_center[0] - cx
                to_center_y = self._image_center[1] - cy
                tan1_x = -ny
                tan1_y = nx   # 左旋
                tan2_x = ny
                tan2_y = -nx  # 右旋
                dot1 = tan1_x * to_center_x + tan1_y * to_center_y
                dot2 = tan2_x * to_center_x + tan2_y * to_center_y
                if dot1 > dot2:
                    tan_x += cfg.apf_k_tangent * mag * tan1_x
                    tan_y += cfg.apf_k_tangent * mag * tan1_y
                else:
                    tan_x += cfg.apf_k_tangent * mag * tan2_x
                    tan_y += cfg.apf_k_tangent * mag * tan2_y

        # 3. 合力
        force_x = att_x + rep_x + tan_x
        force_y = att_y + rep_y + tan_y
        force_mag = math.sqrt(force_x * force_x + force_y * force_y)

        # 4. 局部最小值检测与逃逸
        if force_mag < cfg.apf_stuck_threshold and has_obstacle:
            self._stuck_counter += 1
            if self._stuck_counter >= cfg.apf_stuck_frames:
                # 随机逃逸方向
                import random
                angle = random.uniform(0, 2 * math.pi)
                self._stuck_direction = (
                    cfg.apf_escape_k * math.cos(angle),
                    cfg.apf_escape_k * math.sin(angle),
                )
                self._stuck_counter = 0
            force_x += self._stuck_direction[0]
            force_y += self._stuck_direction[1]
        else:
            self._stuck_counter = max(0, self._stuck_counter - 1)
            # 衰减逃逸力
            self._stuck_direction = (
                self._stuck_direction[0] * 0.9,
                self._stuck_direction[1] * 0.9,
            )

        # 5. 坐标修正
        new_cx = cx + cfg.apf_gain * force_x
        new_cy = cy + cfg.apf_gain * force_y

        # 6. 裁剪到有效图像范围 [0, 320] × [0, 240]
        new_cx = max(0.0, min(320.0, new_cx))
        new_cy = max(0.0, min(240.0, new_cy))

        return (new_cx, new_cy)

    def reset(self) -> None:
        """重置 stuck 计数器。"""
        self._stuck_counter = 0
        self._stuck_direction = (0.0, 0.0)
