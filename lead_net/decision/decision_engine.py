"""避障决策引擎 —— 风险评估器架构（最终版）。

架构（重构后）：
    检测 → 目标管理(Kalman+生命周期) → 风险计算 → 状态机 → 串口输出

五层流程：
    第一层：置信度过滤
    第二层：ROI 空间过滤
    第三层：危险分组 + 面积阈值判定
    第四层：Kalman 多目标追踪（TTL + 遮挡恢复）
    第五层：统一风险评分 + EMA 时间平滑 + 状态机

核心优化（用户建议，2026-07-15）：
    - 时间一致性：EMA 平滑，3帧连续高风险才输出
    - 面积变化率：增长率越高，危险程度越高
    - 统一风险评分：可扩展权重公式，新增指标只需加一项
    - 目标生命周期：TTL > 0 时即使丢失也继续 Kalman 预测
    - 遮挡恢复：lost track 仍输出预测位置，STM32 不突然失去目标
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .roi_filter import ROIFilter, ROIParams
from .priority import PriorityCalculator, PriorityParams
from .fusion import FusionStrategy, FusionParams
from .risk import RiskCalculator, RiskParams, TargetRisk


@dataclass
class DecisionResult:
    """决策引擎最终输出。

    简化输出 (x, y, a)，a = w × h。
    """

    x: int
    y: int
    a: int
    risk_score: float = 0.0
    state: str = "none"       # none|detected|tracking|warning|danger|lost
    source: str = "DL"
    class_id: int = -1

    @classmethod
    def no_target(cls) -> "DecisionResult":
        return cls(x=-1, y=-1, a=0, risk_score=0.0, state="none")

    @property
    def has_target(self) -> bool:
        return self.x >= 0

    def to_uart(self) -> str:
        return f"x{self.x},y{self.y},a{self.a}\r\n"


@dataclass
class DecisionParams:
    """决策引擎总参数。"""

    confidence_threshold: float = 0.5
    roi: ROIParams = field(default_factory=ROIParams)
    priority: PriorityParams = field(default_factory=PriorityParams)
    risk: RiskParams = field(default_factory=RiskParams)
    fusion: FusionParams = field(default_factory=FusionParams)

    # 追踪
    max_targets: int = 3
    min_hits: int = 2
    T_lost: int = 3
    max_ttl: int = 5
    iou_threshold: float = 0.3

    # 时间一致性：需要多少连续帧高风险才稳定输出
    min_consecutive_high_risk: int = 3

    @classmethod
    def from_cfg(cls, cfg: dict) -> "DecisionParams":
        d = cfg.get("decision", {})
        t = cfg.get("tracking", {})
        return cls(
            confidence_threshold=d.get("confidence_threshold", 0.5),
            roi=ROIParams.from_cfg(cfg),
            priority=PriorityParams.from_cfg(cfg),
            risk=RiskParams.from_cfg(cfg),
            fusion=FusionParams.from_cfg(cfg),
            max_targets=t.get("max_targets", 3),
            min_hits=t.get("min_hits", 2),
            T_lost=t.get("T_lost", 3),
            max_ttl=t.get("max_ttl", 5),
            iou_threshold=t.get("iou_threshold", 0.3),
            min_consecutive_high_risk=d.get("min_consecutive_high_risk", 3),
        )


class DecisionEngine:
    """风险评估器 —— 五层决策 + EMA + 状态机。

    用法::

        from lead_net.tracking import MultiTargetTracker

        tracker = MultiTargetTracker(max_targets=3, max_ttl=5)
        engine = DecisionEngine(params, tracker)
        result = engine.decide(dl_detections, cv_regions)
        uart.write(result.to_uart())
    """

    def __init__(
        self,
        params: DecisionParams | None = None,
        tracker: Any = None,
        image_width: int = 320,
        image_height: int = 320,
    ):
        self.params = params or DecisionParams()
        self.image_width = image_width
        self.image_height = image_height

        self._roi_filter = ROIFilter(self.params.roi)
        self._priority_calc = PriorityCalculator(self.params.priority)
        self._risk_calc = RiskCalculator(
            self.params.risk,
            image_center=(image_width / 2, image_height / 2),
            image_area=float(image_width * image_height),
        )
        self._fusion = FusionStrategy(self.params.fusion)
        self._tracker = tracker

        # 输出稳定性：连续高风险帧计数
        self._consecutive_high_risk: int = 0
        self._last_output_state: str = "none"
        self._last_output_track_id: int = -1

        self.stats: dict[str, int] = {
            "total_frames": 0,
            "decisions_made": 0,
            "warnings": 0,
            "dangers": 0,
            "lost_recoveries": 0,
            "layer1_filtered": 0,
            "layer2_filtered": 0,
        }

    # ---- public API ----

    def decide(
        self,
        dl_detections: list[dict[str, Any]],
        cv_regions: list[dict[str, Any]] | None = None,
    ) -> DecisionResult:
        """执行完整风险评估流程。"""
        self.stats["total_frames"] += 1

        # ---- 融合：DL + CV ----
        if cv_regions:
            candidates = self._fusion.merge(
                dl_detections, cv_regions,
                self.image_width, self.image_height,
            )
        else:
            candidates = [dict(d) for d in dl_detections]
            for c in candidates:
                c.setdefault("source", "DL")

        # ==== L1：置信度过滤 ====
        before = len(candidates)
        candidates = [c for c in candidates if c.get("score", 0) >= self.params.confidence_threshold]
        self.stats["layer1_filtered"] += before - len(candidates)
        if not candidates:
            return self._handle_no_detection()

        # ==== L2：ROI ====
        before = len(candidates)
        candidates, _ = self._roi_filter.apply(candidates, self.image_width, self.image_height)
        self.stats["layer2_filtered"] += before - len(candidates)
        if not candidates:
            return self._handle_no_detection()

        # ==== L3：分组面积阈值 ====
        urgent, tracked = self._priority_calc.classify(candidates)

        # ==== L4：Kalman 追踪 ====
        if self._tracker is not None:
            all_for_tracking = urgent + tracked
            self._tracker.update(all_for_tracking)
            confirmed = self._tracker.confirmed()
            lost = self._tracker.get_lost_predictions()

            # 遮挡恢复：检查 lost tracks 是否被重新匹配
            if lost:
                self.stats["lost_recoveries"] += len(lost)

            # 合并 confirmed + lost（lost 仍可预测位置）
            active_tracks = confirmed + lost
        else:
            active_tracks = urgent

        # ==== L5：风险评分 + 状态机 ====
        if active_tracks:
            result = self._evaluate_risks(active_tracks)
        else:
            result = self._handle_no_detection()

        # 清理无效目标历史
        if self._tracker is not None:
            active_ids = {t.id for t in active_tracks}
            self._risk_calc.prune(active_ids)

        return result

    def reset_stats(self) -> None:
        for k in self.stats:
            self.stats[k] = 0
        self._consecutive_high_risk = 0

    # ---- internal ----

    def _evaluate_risks(self, tracks: list[Any]) -> DecisionResult:
        """对活跃 tracks 进行风险评分，选出最高风险目标。"""
        best_score = -1.0
        best_track = None
        best_risk = None

        for t in tracks:
            if hasattr(t, "kf"):
                s = t.kf.state()
                cx, cy, w, h = float(s[0]), float(s[1]), float(s[2]), float(s[3])
                tid = t.id
                conf = t.conf
            else:
                bbox = t["bbox"]
                cx, cy, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                tid = t.get("id", -1)
                conf = t.get("score", 0.0)

            area = w * h
            risk = self._risk_calc.evaluate(tid, area, conf, cx, cy)

            # 更新 Track 决策状态
            if hasattr(t, "decision_state"):
                t.decision_state = risk.state

            if risk.smoothed_risk > best_score:
                best_score = risk.smoothed_risk
                best_track = t
                best_risk = risk

        if best_track is None or best_risk is None:
            return self._handle_no_detection()

        # 时间一致性：需要连续 min_consecutive_high_risk 帧高风险才稳定输出
        if best_risk.state in ("warning", "danger"):
            if best_track is not None and getattr(best_track, "id", -1) == self._last_output_track_id:
                self._consecutive_high_risk += 1
            else:
                self._consecutive_high_risk = 1
            self._last_output_track_id = getattr(best_track, "id", -1)
        else:
            self._consecutive_high_risk = max(0, self._consecutive_high_risk - 1)

        # 连续高风险帧不足 → 降低输出（但跟踪中仍可输出 tracking 级别）
        if best_risk.state == "danger" and self._consecutive_high_risk < max(1, self.params.min_consecutive_high_risk // 2):
            pass  # danger 级别 threshold 减半，更快响应
        elif best_risk.state == "warning" and self._consecutive_high_risk < self.params.min_consecutive_high_risk:
            best_risk.state = "tracking"

        # 提取坐标
        if hasattr(best_track, "kf"):
            s = best_track.kf.state()
            cx, cy, w, h = int(s[0]), int(s[1]), int(s[2]), int(s[3])
            cls_id = best_track.cls
            source = "DL"
        else:
            bbox = best_track["bbox"]
            cx, cy, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cls_id = best_track.get("category_id", -1)
            source = best_track.get("source", "DL")

        self.stats["decisions_made"] += 1
        if best_risk.state == "warning":
            self.stats["warnings"] += 1
        elif best_risk.state == "danger":
            self.stats["dangers"] += 1

        self._last_output_state = best_risk.state
        return DecisionResult(
            x=cx, y=cy, a=int(w * h),
            risk_score=best_risk.smoothed_risk,
            state=best_risk.state,
            source=source,
            class_id=cls_id,
        )

    def _handle_no_detection(self) -> DecisionResult:
        """无检测/无风险目标时的输出。"""
        self._consecutive_high_risk = max(0, self._consecutive_high_risk - 1)
        # 如果最近曾输出过目标，继续保持短暂输出（避免 STM32 突然失去目标）
        if self._last_output_state not in ("none",) and self._consecutive_high_risk > 0:
            return DecisionResult(x=-1, y=-1, a=0, risk_score=0.0, state="lost")
        self._last_output_state = "none"
        return DecisionResult.no_target()
