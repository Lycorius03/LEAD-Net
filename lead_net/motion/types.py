"""运动规划模块 — 共享类型定义（v4: 五状态FSM + 目标重拾）。

MotionPlanner 操作在 model space (320×320) 中，输出仍是 DecisionResult。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PlannerState(Enum):
    """行为状态机状态枚举（v4: 五状态追踪-避障-重拾架构）。

    SEARCHING → TRACKING → OBSTACLE_DETECTED → AVOIDING → TARGET_REACQUIRE
    """
    SEARCHING = "searching"               # 初始/全局搜索目标
    TRACKING = "tracking"                 # 正常追踪（含减速子状态）
    OBSTACLE_DETECTED = "obstacle_detected"  # 发现障碍，评估是否需要绕行
    AVOIDING = "avoiding"                 # 绕障中
    TARGET_REACQUIRE = "target_reacquire" # 目标丢失/遮挡后重拾搜索


class FsmEvent(Enum):
    """触发状态转换的事件（v4: 新增障碍相关事件 + 重拾事件）。"""
    TARGET_FOUND = "target_found"              # 检测到目标
    TARGET_LOST = "target_lost"                # 目标暂时丢失（TTL>0）
    TARGET_LOST_TTL0 = "target_lost_ttl0"      # TTL 耗尽，彻底丢失
    OBSTACLE_NEAR = "obstacle_near"            # 前方检测到障碍物
    OBSTACLE_NOT_THREAT = "obstacle_not_threat" # 障碍不威胁，继续追踪
    OBSTACLE_CLEARED = "obstacle_cleared"       # 障碍物已清除
    TARGET_CLOSE = "target_close"               # 目标面积过大（距离太近）
    AREA_NORMAL = "area_normal"                 # 目标面积恢复正常
    TARGET_REACQUIRED = "target_reacquired"     # 重拾成功，恢复追踪
    REACQUIRE_TIMEOUT = "reacquire_timeout"     # 重拾超时，回到全局搜索


@dataclass
class MotionConfig:
    """运动规划器总配置（v4: 增强重拾参数）。

    所有权重和阈值可配置，通过 configs/*.yaml 传入。
    数值为默认值，需在硬件上实测校准。
    """

    # ---- APF 避障 ----
    apf_k_rep: float = 1.0
    apf_k_att: float = 0.5
    apf_d0: float = 100.0
    apf_k_tangent: float = 0.3
    apf_gain: float = 0.5
    apf_escape_k: float = 2.0
    apf_stuck_threshold: float = 1.0
    apf_stuck_frames: int = 5

    # ---- 速度控制（v4: 五级精细映射） ----
    speed_target_area: float = 5000.0
    speed_area_fast: float = 0.05      # <5% 面积 → 高速
    speed_area_medium: float = 0.15    # 5-15% → 中速
    speed_area_slow: float = 0.20      # 15-20% → 慢速
    speed_area_very_slow: float = 0.30 # 20-30% → 极慢, >30% → 停止
    speed_scale_fast: float = 1.0
    speed_scale_medium: float = 0.7
    speed_scale_slow: float = 0.4
    speed_scale_very_slow: float = 0.2
    speed_scale_stop: float = 0.0
    speed_ema_beta: float = 0.4
    speed_cy_pushback: float = 10.0

    # ---- 路径记忆 ----
    memory_max_len: int = 10
    memory_search_radius_start: int = 20
    memory_search_radius_step: int = 10
    memory_search_radius_max: int = 100
    memory_golden_angle: float = 137.5

    # ---- 轨迹预测 ----
    predict_short_occlusion: int = 3
    predict_max_ttl: int = 15
    predict_kalman_steps_max: int = 10

    # ---- 障碍物判定 ----
    obstacle_iou_max: float = 0.1
    obstacle_dist_min: float = 80.0
    obstacle_min_area: float = 500.0

    # ---- 状态机 ----
    fsm_min_state_frames: int = 3
    fsm_approach_cooldown: int = 10

    # ---- 重拾机制（v4 新增） ----
    reacquire_max_frames: int = 90        # 重拾最大帧数（~3秒@30fps）
    reacquire_roi_expand_rate: float = 0.15  # ROI 每帧扩展比例
    reacquire_roi_max_expand: float = 2.0    # ROI 最大扩展倍数
    reacquire_spiral_a: float = 5.0          # 螺旋参数 a (px)
    reacquire_spiral_b: float = 3.0          # 螺旋参数 b (px/frame)
    reacquire_confidence_boost: float = 0.8  # 重拾时置信度阈值降低系数

    @classmethod
    def from_cfg(cls, cfg: dict) -> "MotionConfig":
        """从 YAML 配置构造 MotionConfig。"""
        mc = cfg.get("motion", {})
        apf = mc.get("apf", {})
        spd = mc.get("speed", {})
        mem = mc.get("memory", {})
        pred = mc.get("predict", {})
        obs = mc.get("obstacle", {})
        fsm = mc.get("fsm", {})
        reacq = mc.get("reacquire", {})
        return cls(
            apf_k_rep=apf.get("k_rep", 1.0),
            apf_k_att=apf.get("k_att", 0.5),
            apf_d0=apf.get("d0", 100.0),
            apf_k_tangent=apf.get("k_tangent", 0.3),
            apf_gain=apf.get("gain", 0.5),
            apf_escape_k=apf.get("escape_k", 2.0),
            apf_stuck_threshold=apf.get("stuck_threshold", 1.0),
            apf_stuck_frames=apf.get("stuck_frames", 5),
            speed_target_area=spd.get("target_area", 5000.0),
            speed_area_fast=spd.get("area_fast", 0.05),
            speed_area_medium=spd.get("area_medium", 0.15),
            speed_area_slow=spd.get("area_slow", 0.20),
            speed_area_very_slow=spd.get("area_very_slow", 0.30),
            speed_scale_fast=spd.get("scale_fast", 1.0),
            speed_scale_medium=spd.get("scale_medium", 0.7),
            speed_scale_slow=spd.get("scale_slow", 0.4),
            speed_scale_very_slow=spd.get("scale_very_slow", 0.2),
            speed_scale_stop=spd.get("scale_stop", 0.0),
            speed_ema_beta=spd.get("ema_beta", 0.4),
            speed_cy_pushback=spd.get("cy_pushback", 10.0),
            memory_max_len=mem.get("max_len", 10),
            memory_search_radius_start=mem.get("search_radius_start", 20),
            memory_search_radius_step=mem.get("search_radius_step", 10),
            memory_search_radius_max=mem.get("search_radius_max", 100),
            memory_golden_angle=mem.get("golden_angle", 137.5),
            predict_short_occlusion=pred.get("short_occlusion", 3),
            predict_max_ttl=pred.get("max_ttl", 15),
            predict_kalman_steps_max=pred.get("kalman_steps_max", 10),
            obstacle_iou_max=obs.get("iou_max", 0.1),
            obstacle_dist_min=obs.get("dist_min", 80.0),
            obstacle_min_area=obs.get("min_area", 500.0),
            fsm_min_state_frames=fsm.get("min_state_frames", 3),
            fsm_approach_cooldown=fsm.get("approach_cooldown", 10),
            reacquire_max_frames=reacq.get("max_frames", 90),
            reacquire_roi_expand_rate=reacq.get("roi_expand_rate", 0.15),
            reacquire_roi_max_expand=reacq.get("roi_max_expand", 2.0),
            reacquire_spiral_a=reacq.get("spiral_a", 5.0),
            reacquire_spiral_b=reacq.get("spiral_b", 3.0),
            reacquire_confidence_boost=reacq.get("confidence_boost", 0.8),
        )
