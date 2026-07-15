"""避障决策引擎 —— 风险评估器架构（最终版）。

端到端数据流（完整）：:

      Camera Frame (e.g. 320×240 物理摄像头)
          │
          ├── DL Preprocessor (resize/letterbox → 320×320) ──→ [LEAD-Net]
          │                                                        │
          │    dl_detections in 320×320 model space                │
          │    [{"bbox":[cx,cy,w,h], "score":, "category_id":}, …] │
          │                                                        │
          └── [CV Fallback] (ground seg → blob detect → normalize) │
                   │                                                │
                   │  cv_regions in 320×320 model space            │
                   │  [{"bbox":[cx,cy,w,h], "score":,               │
                   │    "area":w×h, "source":"CV"}, …]             │
                   │                                                │
                   ▼                                                ▼
          [FusionStrategy.merge()]  ── IoU matching in unified 320×320
                   │
                   ▼
          [DecisionEngine.decide()]  5-layer pipeline (all in 320×320)
                   │
                   │  L1: Confidence filter   L2: ROI spatial filter
                   │  L3: Priority classifier  L4: Kalman Multi-Target Tracker
                   │  L5: Risk scoring (EMA + growth rate + state machine)
                   │
                   ▼
          [DecisionResult]  ── internal state in 320×320 model space
              │ .x, .y, .w, .h, .a (= w×h), .risk_score, .state, .source
              │
              ├── .to_uart(target_w=320, target_h=240, model_w=320, model_h=320)
              │       scales: 320×320 → 320×240 (STM32 physical coordinate system)
              │       outputs: "x{cx},y{cy},a{area}\\r\\n"  or  "x-1,y-1,a0\\r\\n"
              │       → STM32 via UART at 20 Hz
              │
              └── USBLogger.format_row()
                      outputs: "timestamp,frame,fps,class,conf,cx,cy,w,h,area,track_id\\n"
                      → Host PC via USB/VCP for debug & experiment analysis

坐标空间约定:
    - Model space:  320×320 (square, what DL model sees after resize)
    - Target space: 320×240 (what STM32 expects, matches physical camera)
    - CV fallback normalizes from raw camera resolution → model space 320×320
    - to_uart() converts from model space → target space (STM32 physical)
    - Area = w × h (bbox 面积, NOT blob pixel count), 在各自坐标空间中计算
    - 丢失标志: x=-1, y=-1, a=0 (与 STM32 现有协议 100% 兼容)

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
class TargetState:
    """Unified internal target state in model space (e.g., 320x320)."""

    track_id: int
    class_id: int
    confidence: float
    cx: float
    cy: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    state: str = "none"       # none|detected|tracking|warning|danger|lost
    source: str = "DL"


class USBLogger:
    """Formats rich VCP/USB logs for host-side experimentation and analysis."""

    @staticmethod
    def header() -> str:
        return "timestamp,frame,fps,class,conf,cx,cy,w,h,area,track_id\n"

    @staticmethod
    def format_row(timestamp_ms: int, frame_id: int, fps: float, target: TargetState) -> str:
        # Standard 7-class map name lookup
        class_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "backpack",
            4: "suitcase", 5: "chair", 6: "bottle"
        }
        class_name = class_names.get(target.class_id, f"cls_{target.class_id}")
        area = int(round(target.w * target.h))
        return (
            f"{timestamp_ms},"
            f"{frame_id},"
            f"{fps:.1f},"
            f"{class_name},"
            f"{target.confidence:.2f},"
            f"{target.cx:.1f},"
            f"{target.cy:.1f},"
            f"{target.w:.1f},"
            f"{target.h:.1f},"
            f"{area},"
            f"{target.track_id}\n"
        )


@dataclass
class DecisionResult:
    """决策引擎最终输出 —— UART 协议转换层。

    内部使用 model space (320×320)，通过 to_uart() 缩放到
    STM32 物理坐标系 (e.g. 320×240)。

    Protocol:
        有目标: "x{cx},y{cy},a{area}\\r\\n"
        丢失:   "x-1,y-1,a0\\r\\n"
        其中 area = bbox_width × bbox_height (像素²)，在 to_uart() 中
        按 target/model 比例缩放。
    """

    x: int
    y: int
    a: int
    w: int = 0
    h: int = 0
    risk_score: float = 0.0
    state: str = "none"       # none|detected|tracking|warning|danger|lost
    source: str = "DL"
    class_id: int = -1

    @classmethod
    def no_target(cls) -> "DecisionResult":
        return cls(x=-1, y=-1, a=0, w=0, h=0, risk_score=0.0, state="none")

    @classmethod
    def from_target_state(cls, target: TargetState, risk_score: float) -> "DecisionResult":
        return cls(
            x=int(round(target.cx)),
            y=int(round(target.cy)),
            a=int(round(target.w * target.h)),
            w=int(round(target.w)),
            h=int(round(target.h)),
            risk_score=risk_score,
            state=target.state,
            source=target.source,
            class_id=target.class_id,
        )

    @property
    def has_target(self) -> bool:
        return self.x >= 0

    def to_uart(self, target_w: int = 320, target_h: int = 240,
                model_w: int = 320, model_h: int = 320) -> str:
        """格式化输出串口数据，包含坐标和面积缩放以对齐 STM32 预期的物理坐标系 (e.g. 320x240)。"""
        if self.x == -1 and self.y == -1:
            return "x-1,y-1,a0\r\n"
        
        # 坐标投影
        scaled_x = int(round(self.x * target_w / model_w))
        scaled_y = int(round(self.y * target_h / model_h))
        
        # 面积缩放
        if self.w > 0 and self.h > 0:
            scaled_w = self.w * target_w / model_w
            scaled_h = self.h * target_h / model_h
            scaled_a = int(round(scaled_w * scaled_h))
        else:
            scaled_a = int(round(self.a * (target_w * target_h) / (model_w * model_h)))
            
        return f"x{scaled_x},y{scaled_y},a{scaled_a}\r\n"


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

        # 提取坐标和状态以创建统一的 TargetState
        if hasattr(best_track, "kf"):
            s = best_track.kf.state()
            cx, cy, w, h = float(s[0]), float(s[1]), float(s[2]), float(s[3])
            vx = float(s[4])
            vy = float(s[5])
            cls_id = best_track.cls
            source = "DL"
        else:
            bbox = best_track["bbox"]
            cx, cy, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            vx, vy = 0.0, 0.0
            cls_id = best_track.get("category_id", -1)
            source = best_track.get("source", "DL")

        target = TargetState(
            track_id=getattr(best_track, "id", -1),
            class_id=cls_id,
            confidence = float(best_track.conf if hasattr(best_track, "conf") else best_track.get("score", 0.0)),
            cx=cx,
            cy=cy,
            w=w,
            h=h,
            vx=vx,
            vy=vy,
            state=best_risk.state,
            source=source
        )

        self.stats["decisions_made"] += 1
        if best_risk.state == "warning":
            self.stats["warnings"] += 1
        elif best_risk.state == "danger":
            self.stats["dangers"] += 1

        self._last_output_state = best_risk.state
        return DecisionResult.from_target_state(target, best_risk.smoothed_risk)

    def _handle_no_detection(self) -> DecisionResult:
        """无检测/无风险目标时的输出。"""
        self._consecutive_high_risk = max(0, self._consecutive_high_risk - 1)
        # 如果最近曾输出过目标，继续保持短暂输出（避免 STM32 突然失去目标）
        if self._last_output_state not in ("none",) and self._consecutive_high_risk > 0:
            return DecisionResult(x=-1, y=-1, a=0, risk_score=0.0, state="lost")
        self._last_output_state = "none"
        return DecisionResult.no_target()
