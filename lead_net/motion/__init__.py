"""LEAD-Net 运动规划模块（v4: 五状态FSM + 重拾机制）。

提供自主避障、自适应减速、稳定跟随、目标重拾四大能力。
所有逻辑在 OpenMV 端，UART 协议不变。

用法::

    from lead_net.motion import MotionPlanner, MotionConfig, ReacquisitionEngine

    config = MotionConfig.from_cfg(cfg)
    planner = MotionPlanner(config)
    reacq = ReacquisitionEngine(config)

    result = engine.decide(dl_detections, cv_regions)
    planned = planner.plan(result, all_detections, all_tracks, target_track)
    uart.write(planned.to_uart())
"""

from .types import MotionConfig, PlannerState, FsmEvent
from .behavior_fsm import BehaviorFSM
from .obstacle_avoider import ObstacleAvoider
from .speed_controller import SpeedController
from .path_memory import PathMemory, TrajectoryPredictor
from .reacquisition import ReacquisitionEngine, is_point_in_roi, compute_iou_with_roi
from .motion_planner import MotionPlanner

__all__ = [
    "MotionPlanner",
    "MotionConfig",
    "BehaviorFSM",
    "ObstacleAvoider",
    "SpeedController",
    "PathMemory",
    "TrajectoryPredictor",
    "ReacquisitionEngine",
    "is_point_in_roi",
    "compute_iou_with_roi",
    "PlannerState",
    "FsmEvent",
]
