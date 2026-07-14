"""避障决策引擎 —— 三层决策流程编排。

三层结构（依次执行）：
    第一层：置信度过滤 → 移除低置信度框
    第二层：ROI 空间过滤 → 仅保留避障关注区内的目标
    第三层：面积代理 + 紧急度优先级 → 计算 priority，选 top-1

融合前置（可选）：
    DL/CV 混合 → DL 检测结果与传统 CV 前景掩膜做 IoU 重叠验证

输出：
    DecisionResult = (x, y, w, h, priority, class_id, source)
    或 None（无足够紧急目标）

参考：
    - ADOS (IEEE IV 2024): "On Road Object" score 区分可行驶区域上的未知物体
    - DECADE (2024): 纯 bbox 几何推理避障，不生成像素级深度图
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .roi_filter import ROIFilter, ROIParams
from .priority import PriorityCalculator, PriorityParams
from .fusion import FusionStrategy, FusionParams


@dataclass
class DecisionResult:
    """决策引擎输出。

    Attributes:
        x, y, w, h: 目标 bbox（cxcywh，像素坐标）
        priority: 紧急度分数 (0.0-1.0)
        class_id: 类别 ID（-1=UNKNOWN）
        confidence: 检测置信度
        source: 来源标记 "DL" | "CV" | "FUSION"
        layer_log: 各层通过情况 {"confidence": bool, "roi": bool, "priority": bool}
    """

    x: float
    y: float
    w: float
    h: float
    priority: float
    class_id: int = -1
    confidence: float = 0.0
    source: str = "DL"
    layer_log: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [self.x, self.y, self.w, self.h],
            "priority": self.priority,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "source": self.source,
            "layer_log": self.layer_log,
        }


@dataclass
class DecisionParams:
    """决策引擎总参数（聚合各层参数）。"""

    confidence_threshold: float = 0.3
    roi: ROIParams = field(default_factory=ROIParams)
    priority: PriorityParams = field(default_factory=PriorityParams)
    fusion: FusionParams = field(default_factory=FusionParams)

    @classmethod
    def from_cfg(cls, cfg: dict) -> "DecisionParams":
        d = cfg.get("decision", {})
        return cls(
            confidence_threshold=d.get("confidence_threshold", 0.3),
            roi=ROIParams.from_cfg(cfg),
            priority=PriorityParams.from_cfg(cfg),
            fusion=FusionParams.from_cfg(cfg),
        )


class DecisionEngine:
    """三层避障决策引擎。

    用法::

        engine = DecisionEngine(params)
        result = engine.decide(dl_detections)  # DL-only 模式
        result = engine.decide(dl_detections, cv_regions)  # DL+CV 融合模式

    Attributes:
        stats: 累计统计 {"total_frames":, "decisions_made":, "layer1_filtered":, ...}
    """

    def __init__(
        self,
        params: DecisionParams | None = None,
        image_width: int = 320,
        image_height: int = 320,
    ):
        self.params = params or DecisionParams()
        self.image_width = image_width
        self.image_height = image_height

        image_area = float(image_width * image_height)
        self._roi_filter = ROIFilter(self.params.roi)
        self._priority_calc = PriorityCalculator(self.params.priority,
                                                  image_area=image_area)
        self._fusion = FusionStrategy(self.params.fusion)

        # 运行统计
        self.stats: dict[str, int] = {
            "total_frames": 0,
            "decisions_made": 0,
            "layer1_filtered": 0,
            "layer2_filtered": 0,
            "layer3_filtered": 0,
            "cv_contributions": 0,
        }

    # ---- public API ----

    def decide(
        self,
        dl_detections: list[dict[str, Any]],
        cv_regions: list[dict[str, Any]] | None = None,
    ) -> DecisionResult | None:
        """执行三层决策，返回最紧急目标或 None。

        Args:
            dl_detections: LEAD-Net decode 输出 [{"bbox":[cx,cy,w,h], "score":, "category_id":}, ...]
            cv_regions: 传统 CV 前景掩膜（可选），格式同 dl_detections

        Returns:
            DecisionResult 或 None
        """
        self.stats["total_frames"] += 1
        layer_log: dict[str, bool] = {}

        # ---- 融合前置：DL + CV ----
        if cv_regions:
            candidates = self._fusion.merge(
                dl_detections, cv_regions,
                self.image_width, self.image_height,
            )
        else:
            candidates = [dict(d) for d in dl_detections]
            for c in candidates:
                c.setdefault("source", "DL")

        # ---- 第一层：置信度过滤 ----
        before_l1 = len(candidates)
        candidates = [
            c for c in candidates
            if c.get("score", 0) >= self.params.confidence_threshold
        ]
        layer_log["confidence"] = len(candidates) > 0
        self.stats["layer1_filtered"] += before_l1 - len(candidates)

        if not candidates:
            return None

        # ---- 第二层：ROI 空间过滤 ----
        before_l2 = len(candidates)
        candidates, _ = self._roi_filter.apply(
            candidates, self.image_width, self.image_height,
        )
        layer_log["roi"] = len(candidates) > 0
        self.stats["layer2_filtered"] += before_l2 - len(candidates)

        if not candidates:
            return None

        # ---- 第三层：面积代理 + 紧急度优先级 ----
        before_l3 = len(candidates)
        scored = self._priority_calc.compute(candidates)
        top = self._priority_calc.select_top(scored)
        layer_log["priority"] = top is not None
        self.stats["layer3_filtered"] += before_l3 - (1 if top else 0)

        if top is None:
            return None

        # 统计 CV 贡献
        if top.get("source") in ("CV", "FUSION"):
            self.stats["cv_contributions"] += 1

        self.stats["decisions_made"] += 1

        bbox = top["bbox"]
        return DecisionResult(
            x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3],
            priority=top.get("priority", 0.0),
            class_id=top.get("category_id", -1),
            confidence=top.get("score", 0.0),
            source=top.get("source", "DL"),
            layer_log=layer_log,
        )

    def reset_stats(self) -> None:
        """重置运行统计。"""
        for k in self.stats:
            self.stats[k] = 0
