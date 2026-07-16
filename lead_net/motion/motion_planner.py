"""运动规划器 — 避障 + 减速 + 稳定跟随的总装模块。

数据流:
    DecisionEngine.decide() → DecisionResult
        → MotionPlanner.plan() → DecisionResult (坐标修正后)
        → .to_uart() → STM32

所有逻辑在 OpenMV 端，UART 协议不变。
通过坐标偏置编码避障，面积缩放编码减速。
"""

from __future__ import annotations

from typing import Any

from .types import MotionConfig, PlannerState, FsmEvent
from .behavior_fsm import BehaviorFSM
from .obstacle_avoider import ObstacleAvoider
from .speed_controller import SpeedController
from .path_memory import PathMemory, TrajectoryPredictor


class MotionPlanner:
    """运动规划器 — 在 DecisionEngine 和 UART 输出之间。

    用法::

        from lead_net.motion import MotionPlanner
        from lead_net.decision.decision_engine import DecisionResult

        planner = MotionPlanner(config)
        result: DecisionResult = engine.decide(dl_dets, cv_regions)
        planned = planner.plan(result, all_detections, all_tracks)
        uart.write(planned.to_uart())
    """

    def __init__(self, config: MotionConfig | None = None):
        self._cfg = config or MotionConfig()
        self._fsm = BehaviorFSM(self._cfg)
        self._avoider = ObstacleAvoider(self._cfg)
        self._speed_ctrl = SpeedController(self._cfg)
        self._path_memory = PathMemory(self._cfg)
        self._predictor = TrajectoryPredictor(self._cfg)

        # 障碍物跟踪: {track_id: {cx, cy, w, h, frames_since_seen}}
        self._obstacle_memory: dict[int, dict[str, Any]] = {}

    # ---- 公共 API ----

    def plan(
        self,
        result: Any,  # DecisionResult
        all_detections: list[dict[str, Any]] | None = None,
        all_tracks: list[Any] | None = None,
        target_track: Any = None,  # Track | None
    ) -> Any:
        """执行完整运动规划管线。

        Args:
            result: DecisionEngine 输出的 DecisionResult
            all_detections: 当前帧所有检测（用于识别障碍物）
            all_tracks: 所有活跃 tracks（用于障碍物记忆）
            target_track: 当前选中的追踪 Track（可选，用于获取速度/历史）

        Returns:
            修改后的 DecisionResult（坐标/面积可能已调整）
        """
        if all_detections is None:
            all_detections = []
        if all_tracks is None:
            all_tracks = []

        # ---- 无目标: IDLE ----
        if not getattr(result, "has_target", True) or result.x < 0:
            self._fsm.transition(FsmEvent.TARGET_LOST_TTL0)
            # 检查是否处于 SEARCHING 阶段（还有 TTL）
            if self._fsm.is_searching:
                return self._handle_searching()
            return result  # 返回原始 no_target

        # ---- 有目标: 更新路径记忆 ----
        self._path_memory.record(result.x, result.y)

        # ---- 障碍物识别 ----
        obstacles = self._identify_obstacles(result, all_detections, all_tracks)

        # ---- 事件判定 ----
        event = self._determine_event(result, obstacles, target_track)

        # ---- 状态转换 ----
        self._fsm.transition(event)

        # ---- 按状态执行策略 ----
        state = self._fsm.state

        if state == PlannerState.AVOIDING:
            result = self._apply_avoiding(result, obstacles)
        elif state == PlannerState.APPROACHING:
            result = self._apply_approaching(result)
        elif state == PlannerState.SEARCHING:
            result = self._apply_searching(result, target_track)
        elif state == PlannerState.TRACKING:
            result = self._apply_tracking(result)

        return result

    def reset(self) -> None:
        """重置所有内部状态。"""
        self._fsm.reset()
        self._avoider.reset()
        self._speed_ctrl.reset()
        self._path_memory.reset()
        self._obstacle_memory.clear()

    @property
    def state(self) -> PlannerState:
        return self._fsm.state

    # ---- 内部 ----

    def _identify_obstacles(
        self,
        result: Any,
        detections: list[dict[str, Any]],
        tracks: list[Any],
    ) -> list[dict[str, Any]]:
        """从非追踪目标的检测/track 中识别障碍物。

        判定条件:
            1. 与追踪目标的 IoU < iou_max（不是同一目标）
            2. 距离 < dist_min（在影响范围内）
            3. 面积 > min_area（过滤噪声）
        """
        cfg = self._cfg
        obstacles: list[dict[str, Any]] = []
        tx, ty, tw, th = result.x, result.y, result.w, result.h

        # 从 tracks 中提取障碍物
        for t in (tracks or []):
            tid = getattr(t, "id", -1)
            # 检查是否是追踪目标本身
            if hasattr(t, "kf"):
                s = t.kf.state()
                ox, oy, ow, oh = float(s[0]), float(s[1]), float(s[2]), float(s[3])
            else:
                continue

            # 跳过已丢失的 track
            if getattr(t, "decision_state", "") == "lost":
                continue

            if ow * oh < cfg.obstacle_min_area:
                continue

            iou = self._box_iou(tx, ty, tw, th, ox, oy, ow, oh)
            if iou >= cfg.obstacle_iou_max:
                continue  # 与追踪目标重叠 → 同一目标

            dist = ((tx - ox) ** 2 + (ty - oy) ** 2) ** 0.5
            if dist < cfg.obstacle_dist_min:
                obstacles.append({
                    "cx": ox, "cy": oy, "w": ow, "h": oh,
                    "dist": dist, "iou": iou, "track_id": tid,
                })

        # 从当前检测中补充
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            ox, oy, ow, oh = bbox[0], bbox[1], bbox[2], bbox[3]

            if ow * oh < cfg.obstacle_min_area:
                continue

            iou = self._box_iou(tx, ty, tw, th, ox, oy, ow, oh)
            if iou >= cfg.obstacle_iou_max:
                continue

            dist = ((tx - ox) ** 2 + (ty - oy) ** 2) ** 0.5
            if dist < cfg.obstacle_dist_min:
                # 去重
                if not any(abs(o["cx"] - ox) < 5 and abs(o["cy"] - oy) < 5
                          for o in obstacles):
                    obstacles.append({
                        "cx": ox, "cy": oy, "w": ow, "h": oh,
                        "dist": dist, "iou": iou, "track_id": -1,
                    })

        # 按距离排序（近的优先）
        obstacles.sort(key=lambda o: o["dist"])
        return obstacles

    def _determine_event(
        self,
        result: Any,
        obstacles: list[dict[str, Any]],
        target_track: Any = None,
    ) -> FsmEvent:
        """从当前状态判定触发事件。"""
        current_state = self._fsm.state

        # 有障碍物 → OBSTACLE_NEAR
        if obstacles:
            return FsmEvent.OBSTACLE_NEAR

        # 面积检查
        area = float(result.w * result.h) if result.w > 0 and result.h > 0 else 0.0
        ratio = area / max(self._cfg.speed_target_area, 1.0)

        if current_state == PlannerState.AVOIDING:
            return FsmEvent.OBSTACLE_CLEARED

        if current_state == PlannerState.APPROACHING:
            if ratio <= self._cfg.speed_area_over:
                return FsmEvent.AREA_NORMAL
            return FsmEvent.TARGET_CLOSE  # 仍太近

        if current_state == PlannerState.SEARCHING:
            if result.has_target and result.x >= 0:
                return FsmEvent.TARGET_FOUND
            if self._fsm.search_frames >= self._cfg.predict_max_ttl:
                return FsmEvent.TTL_EXPIRED
            return FsmEvent.TARGET_LOST  # 保持 SEARCHING

        if current_state == PlannerState.IDLE:
            if result.has_target and result.x >= 0:
                return FsmEvent.TARGET_FOUND
            return FsmEvent.TARGET_LOST_TTL0

        # TRACKING 状态
        if ratio > self._cfg.speed_area_brake or ratio > self._cfg.speed_area_over:
            return FsmEvent.TARGET_CLOSE

        return FsmEvent.TARGET_FOUND  # 正常追踪

    def _apply_avoiding(self, result: Any, obstacles: list[dict[str, Any]]) -> Any:
        """AVOIDING: APF 偏置坐标。"""
        cx, cy = self._avoider.apply(
            float(result.x), float(result.y),
            float(result.w), float(result.h),
            obstacles,
        )
        result.x = int(round(cx))
        result.y = int(round(cy))
        result.state = "avoiding"
        return result

    def _apply_approaching(self, result: Any) -> Any:
        """APPROACHING: 面积缩放减速。"""
        area = float(result.w * result.h) if result.w > 0 and result.h > 0 else 0.0
        new_a, new_cx, new_cy, hint = self._speed_ctrl.apply(
            area, float(result.x), float(result.y),
        )
        result.a = int(round(new_a))
        result.x = int(round(new_cx))
        result.y = int(round(new_cy))
        result.state = hint
        return result

    def _apply_tracking(self, result: Any) -> Any:
        """TRACKING: 正常追踪，不做干预。"""
        area = float(result.w * result.h) if result.w > 0 and result.h > 0 else 0.0
        # 仍做轻度 EMA 平滑（不改变坐标系，只平滑面积）
        new_a, _, _, _ = self._speed_ctrl.apply(area, float(result.x), float(result.y))
        # TRACKING 状态下area不变（speed_ctrl只在ratio>1.2时才缩放）
        result.a = int(round(new_a))
        result.state = "tracking"
        return result

    def _apply_searching(self, result: Any, target_track: Any = None) -> Any:
        """SEARCHING: 预测位置 + 螺旋搜索。"""
        step = self._fsm.search_frames
        cfg = self._cfg

        # 优先用 Kalman 速度预测
        if target_track is not None and hasattr(target_track, "kf"):
            s = target_track.kf.state()
            vx, vy = float(s[4]), float(s[5])
            history = getattr(target_track, "history", None)
            if history is not None and len(history) > 0:
                hist_list = list(history)
            else:
                hist_list = [(float(s[0]), float(s[1]), float(s[2]), float(s[3]))]
            cx, cy = self._predictor.predict(hist_list, step + 1, vx, vy)
        else:
            # 无 track 信息: 用路径记忆获取搜索位置
            cx, cy = self._path_memory.get_search_pos(step)

        result.x = int(round(cx))
        result.y = int(round(cy))
        # 预测阶段面积保持最后已知值
        result.state = "searching"
        return result

    def _handle_searching(self) -> Any:
        """无目标但有 TTL 剩余：用路径记忆生成搜索位置。"""
        from lead_net.decision.decision_engine import DecisionResult

        step = self._fsm.search_frames
        cx, cy = self._path_memory.get_search_pos(step)
        last_pos = self._path_memory.last_position

        # 保持上次的面积
        a = 0
        return DecisionResult(
            x=int(round(cx)),
            y=int(round(cy)),
            a=a,
            w=0, h=0,
            risk_score=0.0,
            state="searching",
            source="MOTION",
        )

    @staticmethod
    def _box_iou(
        ax: float, ay: float, aw: float, ah: float,
        bx: float, by: float, bw: float, bh: float,
    ) -> float:
        """cxcywh 格式两个框的 IoU。"""
        ax1, ay1 = ax - aw / 2, ay - ah / 2
        ax2, ay2 = ax + aw / 2, ay + ah / 2
        bx1, by1 = bx - bw / 2, by - bh / 2
        bx2, by2 = bx + bw / 2, by + bh / 2

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter
        return inter / union if union > 1e-8 else 0.0
