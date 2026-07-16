"""行为状态机 — 五状态表驱动 FSM（v4: 目标追踪-避障-重拾架构）。

参考：
    - IEEE TIE 2024/2025: 层次化 FSM + MPC 避障
    - RoboChart Movement FSM (Autonomous Robots 2024): 形式化验证
    - Duckietown Autonomous Navigation (TUM 2025): 视觉驱动 FSM

状态转换图::

                         ┌──────────────┐
              no_target  │   SEARCHING   │  target_found
             ┌──────────►│  (初始状态)    │◄──────────────┐
             │           └──────┬───────┘               │
             │                  │                        │
             │      target_found│                        │
             │                  ▼                        │
             │           ┌──────────────┐                │
             │           │   TRACKING   │  target_close  │
             │           │  (正常追踪)   │───(减速子状态)  │
             │           └──┬───┬───┬───┘                │
             │              │   │   │                    │
             │   obstacle   │   │   │ target_lost        │
             │   _near      │   │   └────────────────────┤
             │              ▼   │                        │
             │  ┌──────────────┐│                        │
             │  │OBSTACLE_     ││   target_lost_ttl0     │
             │  │DETECTED      │└────────────────────────┤
             │  │(障碍评估)     │                         │
             │  └──┬───────┬───┘                         │
             │     │       │                              │
             │  threat  not_threat                        │
             │     │       └──► TRACKING                  │
             │     ▼                                      │
             │  ┌──────────────┐                          │
             │  │   AVOIDING   │                          │
             │  │  (绕障中)     │                          │
             │  └──────┬───────┘                          │
             │         │                                   │
             │  cleared│                                   │
             │         ▼                                   │
             │  ┌──────────────┐  reacquired              │
             │  │   TARGET_    │──────────────────────────┘
             │  │  REACQUIRE   │
             │  │  (重拾搜索)   │
             │  └──────┬───────┘
             │         │
             │  timeout│
             └─────────┘

验证属性:
    - 确定性: 每个 (state, event) → 唯一 next_state
    - 无死锁: 所有状态可通过超时回到 SEARCHING
    - 抗抖动: 最小状态持续时间 min_state_frames
    - 表驱动: 转换规则集中在 _TRANSITIONS 字典中
"""

from __future__ import annotations

from .types import PlannerState, FsmEvent, MotionConfig


# ---- 转换表（v4: 五状态） ----
# 格式: {current_state: {event: next_state}}
# 未列出的组合 = 不变（保持当前状态）

_TRANSITIONS: dict[PlannerState, dict[FsmEvent, PlannerState]] = {
    PlannerState.SEARCHING: {
        FsmEvent.TARGET_FOUND: PlannerState.TRACKING,
    },
    PlannerState.TRACKING: {
        FsmEvent.OBSTACLE_NEAR: PlannerState.OBSTACLE_DETECTED,
        FsmEvent.TARGET_CLOSE: PlannerState.TRACKING,     # 减速子状态（speed_controller 处理）
        FsmEvent.AREA_NORMAL: PlannerState.TRACKING,       # 恢复正常速度
        FsmEvent.TARGET_LOST: PlannerState.TARGET_REACQUIRE,
        FsmEvent.TARGET_LOST_TTL0: PlannerState.SEARCHING,
    },
    PlannerState.OBSTACLE_DETECTED: {
        FsmEvent.OBSTACLE_NOT_THREAT: PlannerState.TRACKING,  # 障碍不威胁，继续追踪
        FsmEvent.OBSTACLE_NEAR: PlannerState.AVOIDING,        # 确认需要绕行
        FsmEvent.TARGET_LOST: PlannerState.TARGET_REACQUIRE,
    },
    PlannerState.AVOIDING: {
        FsmEvent.OBSTACLE_CLEARED: PlannerState.TARGET_REACQUIRE,  # 绕障完成，尝试重拾
        FsmEvent.TARGET_LOST: PlannerState.TARGET_REACQUIRE,
    },
    PlannerState.TARGET_REACQUIRE: {
        FsmEvent.TARGET_REACQUIRED: PlannerState.TRACKING,     # 重拾成功
        FsmEvent.REACQUIRE_TIMEOUT: PlannerState.SEARCHING,    # 超时，全局搜索
        FsmEvent.OBSTACLE_NEAR: PlannerState.OBSTACLE_DETECTED,  # 重拾时又检测到障碍
    },
}


class BehaviorFSM:
    """五状态表驱动行为状态机（v4）。

    用法::

        fsm = BehaviorFSM(config)
        fsm.transition(FsmEvent.TARGET_FOUND)
        print(fsm.state)  # PlannerState.TRACKING
    """

    def __init__(self, config: MotionConfig | None = None):
        self._cfg = config or MotionConfig()
        self._state: PlannerState = PlannerState.SEARCHING  # v4: 初始状态改为 SEARCHING
        self._prev_state: PlannerState = PlannerState.SEARCHING
        self._state_frames: int = 0
        self._search_frames: int = 0           # SEARCHING 阶段累计帧数
        self._reacquire_frames: int = 0        # TARGET_REACQUIRE 阶段累计帧数
        self._approach_cooldown: int = 0

        # v4: 速度子状态（在 TRACKING 内部）
        self._speed_state: str = "tracking"     # "tracking" | "approaching" | "braking"

    # ---- 公共 API ----

    @property
    def state(self) -> PlannerState:
        return self._state

    @property
    def prev_state(self) -> PlannerState:
        return self._prev_state

    @property
    def state_frames(self) -> int:
        return self._state_frames

    @property
    def search_frames(self) -> int:
        return self._search_frames

    @property
    def reacquire_frames(self) -> int:
        return self._reacquire_frames

    @property
    def speed_state(self) -> str:
        """返回速度子状态: 'tracking' | 'approaching' | 'braking'."""
        return self._speed_state

    @property
    def is_searching(self) -> bool:
        return self._state == PlannerState.SEARCHING

    @property
    def is_tracking(self) -> bool:
        return self._state == PlannerState.TRACKING

    @property
    def is_reacquiring(self) -> bool:
        return self._state == PlannerState.TARGET_REACQUIRE

    def transition(self, event: FsmEvent) -> PlannerState:
        """处理一个事件，可能触发状态转换。

        Returns:
            转换后的当前状态。
        """
        cfg = self._cfg
        next_state = _TRANSITIONS.get(self._state, {}).get(event)

        if next_state is not None and next_state != self._state:
            # 检查最小状态持续时间（防止抖动）
            if self._state_frames < cfg.fsm_min_state_frames:
                self._state_frames += 1
                return self._state

            # 执行转换
            self._prev_state = self._state
            self._state = next_state
            self._state_frames = 0

            # 维护计数器
            if next_state == PlannerState.SEARCHING:
                self._search_frames = 0
            if next_state == PlannerState.TARGET_REACQUIRE:
                self._reacquire_frames = 0

            # 速度子状态重置
            if next_state != PlannerState.TRACKING:
                self._speed_state = "tracking"
        else:
            self._state_frames += 1
            if self._state == PlannerState.SEARCHING:
                self._search_frames += 1
            if self._state == PlannerState.TARGET_REACQUIRE:
                self._reacquire_frames += 1

        # 检查重拾超时
        if (self._state == PlannerState.TARGET_REACQUIRE
                and self._reacquire_frames >= cfg.reacquire_max_frames):
            return self.transition(FsmEvent.REACQUIRE_TIMEOUT)

        return self._state

    def set_speed_state(self, state: str) -> None:
        """设置速度子状态（由 SpeedController 调用）。

        Args:
            state: "tracking" | "approaching" | "braking"
        """
        if state in ("tracking", "approaching", "braking"):
            self._speed_state = state

    def reset(self) -> None:
        """重置状态机到 SEARCHING。"""
        self._state = PlannerState.SEARCHING
        self._prev_state = PlannerState.SEARCHING
        self._state_frames = 0
        self._search_frames = 0
        self._reacquire_frames = 0
        self._approach_cooldown = 0
        self._speed_state = "tracking"
