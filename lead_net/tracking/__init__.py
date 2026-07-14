"""LEAD-Net Tracking 模块 —— Kalman 滤波多目标追踪后处理。

对应 PLAN §M4，服务 RQ4（追踪稳定性）。

提供：
    - KalmanFilter：8 维状态匀速模型 Kalman 滤波器
    - MultiTargetTracker：多目标追踪器（贪心 IoU 匹配 + 生命周期管理
      + 优先级选择）
"""

from .kalman_filter import KalmanFilter
from .tracker import MultiTargetTracker, Track

__all__ = ["KalmanFilter", "MultiTargetTracker", "Track"]
