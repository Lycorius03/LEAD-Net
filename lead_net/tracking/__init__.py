"""LEAD-Net Tracking 模块 — Kalman 滤波多目标追踪后处理（v4: NSA-KF + DIOU + MOSSE）。

对应 PLAN §M4，服务 RQ4（追踪稳定性）。

v4 增强:
    - NSA-KF: 噪声自适应 Kalman，提高遮挡后重捕获率
    - DIOU 匹配: 中心距离感知的贪心匹配
    - MOSSE: 轻量相关滤波辅助追踪器（DF 检测失败时桥接）

提供：
    - KalmanFilter：8 维 NSA-KF（自适应噪声）
    - MultiTargetTracker：DIOU 贪心匹配 + 生命周期管理 + 优先级选择
    - MOSSETracker：轻量相关滤波追踪器（仅依赖 numpy）
"""

from .kalman_filter import KalmanFilter
from .tracker import MultiTargetTracker, Track
from .mosse_tracker import MOSSETracker

__all__ = ["KalmanFilter", "MultiTargetTracker", "Track", "MOSSETracker"]
