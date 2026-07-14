"""8 维 Kalman 滤波器（匀速模型），用于单目标帧间平滑。

状态向量 (8,)：
    [cx, cy, w, h, vx, vy, vw, vh]

观测向量 (4,)：
    [cx, cy, w, h]  —— 来自 SSD-Lite 检测头

参考：
    - SORT (arXiv:1602.00763)：Kalman 预测 + IoU 匹配框架
    - 本项目采用 xywh 状态（非 SORT 的 u,v,s,r），直接对应检测头输出
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """8 维匀速 Kalman 滤波器。

    每帧调用 predict() 做状态外推，检测匹配成功后调用 update(z) 做观测修正。
    首次检测直接 init() 初始化状态与协方差。

    Args:
        dt: 帧间时间间隔（假设恒定帧率），默认 1.0。
    """

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None

        # 转移矩阵 F (8x8)：position += velocity * dt
        self._F = np.eye(8, dtype=np.float64)
        self._F[0, 4] = dt
        self._F[1, 5] = dt
        self._F[2, 6] = dt
        self._F[3, 7] = dt

        # 观测矩阵 H (4x8)：取前 4 维
        self._H = np.zeros((4, 8), dtype=np.float64)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._H[3, 3] = 1.0

        # 过程噪声 Q (8x8)：位置噪声小，速度噪声中等
        q_pos = 1.0
        q_vel = 1.0
        self._Q = np.diag([
            q_pos, q_pos, q_pos, q_pos,
            q_vel, q_vel, q_vel, q_vel,
        ]).astype(np.float64)

        # 观测噪声 R (4x4)：检测器不确定性，尺寸噪声大于坐标噪声
        self._R = np.diag([
            5.0 ** 2,   # cx
            5.0 ** 2,   # cy
            10.0 ** 2,  # w (尺寸估计更不稳定)
            10.0 ** 2,  # h
        ]).astype(np.float64)

    # ---- public API ----

    def init(self, cx: float, cy: float, w: float, h: float) -> None:
        """首帧初始化：位置 = 检测值，速度 = 0，协方差默认。"""
        self._x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        p_pos = 10.0 ** 2   # 位置初值方差
        p_vel = 50.0 ** 2   # 速度初值方差（未观测，高不确定）
        self._P = np.diag([
            p_pos, p_pos, p_pos, p_pos,
            p_vel, p_vel, p_vel, p_vel,
        ]).astype(np.float64)

    def predict(self) -> np.ndarray:
        """状态外推（先验估计）。返回预测状态 (8,)。"""
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._x.copy()

    def update(self, z: np.ndarray) -> None:
        """观测修正（后验估计）。z = [cx, cy, w, h] (4,)。"""
        y = z - self._H @ self._x                        # innovation
        S = self._H @ self._P @ self._H.T + self._R      # innovation covariance
        K = self._P @ self._H.T @ np.linalg.inv(S)       # Kalman gain
        self._x = self._x + K @ y
        self._P = (np.eye(8) - K @ self._H) @ self._P

    def state(self) -> np.ndarray:
        """返回当前状态 (8,)：[cx, cy, w, h, vx, vy, vw, vh]."""
        return self._x.copy()

    @property
    def is_initialized(self) -> bool:
        return self._x is not None
