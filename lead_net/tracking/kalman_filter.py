"""8 维 Kalman 滤波器 — NSA (Noise Scale Adaptive) 增强版（v4）。

状态向量 (8,): [cx, cy, w, h, vx, vy, vw, vh]
观测向量 (4,): [cx, cy, w, h]

v4 增强:
    - NSA (Noise Scale Adaptive): 根据 track 置信度动态调整过程噪声 Q
      遮挡期间增大 Q（允许更大不确定性），重检测后减小 Q（快速收敛）
    - 论文依据: Cao et al., IEEE ICECAI 2024 — NA-KF 重捕获率 80.63%

参考:
    - SORT (arXiv:1602.00763): Kalman 预测 + IoU 匹配框架
    - DeepSORT w/ NSA-KF (2024): 自适应噪声 + DIOU
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """8 维匀速 NSA-Kalman 滤波器（v4 增强版）。

    每帧调用 predict() 做状态外推，检测匹配成功后调用 update(z) 做观测修正。
    首次检测直接 init() 初始化状态与协方差。

    Args:
        dt: 帧间时间间隔（假设恒定帧率），默认 1.0。
        nsa_enabled: 是否启用 NSA 自适应噪声，默认 True。
        nsa_k: NSA 噪声缩放系数，Q_adapted = Q_base * (1 + k * (1 - confidence))。
    """

    def __init__(self, dt: float = 1.0, nsa_enabled: bool = True, nsa_k: float = 3.0):
        self.dt = dt
        self.nsa_enabled = nsa_enabled
        self.nsa_k = nsa_k
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None

        # 转移矩阵 F (8x8): position += velocity * dt
        self._F = np.eye(8, dtype=np.float64)
        self._F[0, 4] = dt
        self._F[1, 5] = dt
        self._F[2, 6] = dt
        self._F[3, 7] = dt

        # 观测矩阵 H (4x8): 取前 4 维
        self._H = np.zeros((4, 8), dtype=np.float64)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._H[3, 3] = 1.0

        # 基础过程噪声 Q_base (8x8)
        q_pos = 1.0
        q_vel = 1.0
        self._Q_base = np.diag([
            q_pos, q_pos, q_pos, q_pos,
            q_vel, q_vel, q_vel, q_vel,
        ]).astype(np.float64)
        self._Q = self._Q_base.copy()

        # 观测噪声 R (4x4)
        self._R = np.diag([
            5.0 ** 2,   # cx
            5.0 ** 2,   # cy
            10.0 ** 2,  # w
            10.0 ** 2,  # h
        ]).astype(np.float64)

        self._confidence: float = 1.0

    # ---- NSA 自适应噪声 ----

    def set_confidence(self, confidence: float) -> None:
        """设置当前 track 置信度，自动调整过程噪声 Q。

        Q_adapted = Q_base * (1 + nsa_k * max(0, 1 - confidence))

        Args:
            confidence: ∈ [0, 1]，track 匹配置信度。
        """
        self._confidence = max(0.0, min(1.0, confidence))
        if self.nsa_enabled:
            scale = 1.0 + self.nsa_k * max(0.0, 1.0 - self._confidence)
            self._Q = self._Q_base * scale
        else:
            self._Q = self._Q_base.copy()

    def get_current_q_scale(self) -> float:
        """返回当前 Q 的缩放因子（用于调试/日志）。"""
        if not self.nsa_enabled:
            return 1.0
        return 1.0 + self.nsa_k * max(0.0, 1.0 - self._confidence)

    # ---- public API ----

    def init(self, cx: float, cy: float, w: float, h: float) -> None:
        """首帧初始化: 位置 = 检测值，速度 = 0，协方差默认。"""
        self._x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        p_pos = 10.0 ** 2
        p_vel = 50.0 ** 2
        self._P = np.diag([
            p_pos, p_pos, p_pos, p_pos,
            p_vel, p_vel, p_vel, p_vel,
        ]).astype(np.float64)

        self.set_confidence(1.0)

    def predict(self) -> np.ndarray:
        """状态外推（先验估计）。返回预测状态 (8,)。"""
        if self._x is None:
            raise RuntimeError("KalmanFilter not initialized. Call init() first.")
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._x.copy()

    def predict_only(self, steps: int = 1) -> np.ndarray:
        """纯预测（不更新内部状态），用于多步外推。

        Args:
            steps: 外推步数。

        Returns:
            预测状态 (8,)。
        """
        x = self._x.copy()
        P = self._P.copy()
        for _ in range(steps):
            x = self._F @ x
            P = self._F @ P @ self._F.T + self._Q
        return x

    def update(self, z: np.ndarray, confidence: float | None = None) -> None:
        """观测修正（后验估计）。

        Args:
            z: [cx, cy, w, h] (4,)
            confidence: 当前匹配置信度 ∈ [0,1]，用于 NSA 自适应。
        """
        if confidence is not None:
            self.set_confidence(confidence)

        y = z - self._H @ self._x                        # innovation
        S = self._H @ self._P @ self._H.T + self._R      # innovation covariance
        K = self._P @ self._H.T @ np.linalg.inv(S)       # Kalman gain
        self._x = self._x + K @ y
        self._P = (np.eye(8) - K @ self._H) @ self._P

    def update_confidence_only(self, confidence: float) -> None:
        """仅更新置信度（无观测时调用，增大 Q 以增加不确定性）。

        用于目标暂时丢失但 track 尚未删除期间。
        """
        self.set_confidence(confidence)

    def state(self) -> np.ndarray:
        """返回当前状态 (8,): [cx, cy, w, h, vx, vy, vw, vh]."""
        if self._x is None:
            raise RuntimeError("KalmanFilter not initialized. Call init() first.")
        return self._x.copy()

    @property
    def velocity(self) -> tuple[float, float]:
        """返回当前速度估计 (vx, vy)。"""
        if self._x is None:
            return (0.0, 0.0)
        return (float(self._x[4]), float(self._x[5]))

    @property
    def position(self) -> tuple[float, float, float, float]:
        """返回当前位置估计 (cx, cy, w, h)。"""
        if self._x is None:
            return (0.0, 0.0, 0.0, 0.0)
        return (float(self._x[0]), float(self._x[1]),
                float(self._x[2]), float(self._x[3]))

    @property
    def is_initialized(self) -> bool:
        return self._x is not None

    @property
    def confidence(self) -> float:
        return self._confidence
