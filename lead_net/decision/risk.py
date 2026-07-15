"""统一风险评分器 —— 可扩展的多因素风险评估。

公式（工业标准参考）:
    Risk = w_center × CenterScore + w_area × AreaScore
         + w_growth × GrowthRate  + w_conf  × Confidence

每个分量归一化到 [0, 1]，权重可配置。

时间一致性（EMA 平滑）:
    smoothed[t] = α × raw[t] + (1-α) × smoothed[t-1]
    默认 α=0.3，新检测权重更高，响应更快。

面积变化率（接近速度估计）:
    growth[t] = (area[t] - area[t-1]) / max(area[t-1], 1)
    正值 = 接近中，负值 = 远离中

设计原则:
    - 每个评分项独立计算、独立权重
    - 新增指标只需加一项，整个框架不动
    - O(N) 简单数值计算，OpenMV 兼容
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskParams:
    """风险评分参数（所有权重可配置）。"""

    # 权重（总和不必为 1）
    w_center: float = 0.25
    w_area: float = 0.35
    w_growth: float = 0.25
    w_confidence: float = 0.15

    # EMA 平滑系数（0-1，越大对新数据越敏感）
    ema_alpha: float = 0.3

    # 增长率计算窗口（帧数）
    growth_window: int = 5

    # 危险阈值
    warning_threshold: float = 0.4   # 超过此值→Warning
    danger_threshold: float = 0.7    # 超过此值→Danger

    @classmethod
    def from_cfg(cls, cfg: dict) -> "RiskParams":
        d = cfg.get("decision", {}).get("risk", {})
        return cls(
            w_center=d.get("w_center", 0.25),
            w_area=d.get("w_area", 0.35),
            w_growth=d.get("w_growth", 0.25),
            w_confidence=d.get("w_confidence", 0.15),
            ema_alpha=d.get("ema_alpha", 0.3),
            growth_window=d.get("growth_window", 5),
            warning_threshold=d.get("warning_threshold", 0.4),
            danger_threshold=d.get("danger_threshold", 0.7),
        )


@dataclass
class TargetRisk:
    """单个目标的完整风险评估。"""

    track_id: int
    raw_risk: float = 0.0       # 当前帧原始风险
    smoothed_risk: float = 0.0  # EMA 平滑后风险
    center_score: float = 0.0
    area_score: float = 0.0
    growth_rate: float = 0.0
    confidence: float = 0.0
    last_area: float = 0.0
    state: str = "detected"     # detected | tracking | warning | danger | lost

    def update_state(self) -> str:
        """根据 smoothed_risk 更新状态机。"""
        if self.smoothed_risk >= 0.7:
            self.state = "danger"
        elif self.smoothed_risk >= 0.4:
            self.state = "warning"
        elif self.smoothed_risk > 0.0:
            self.state = "tracking"
        else:
            self.state = "detected"
        return self.state


class RiskCalculator:
    """多因素统一风险评估器。

    用法::

        calc = RiskCalculator(params, image_center, image_area)
        risk = calc.evaluate(track_id, area, confidence, cx, cy, prev_area)
        # risk.smoothed_risk 即最终分数
    """

    def __init__(
        self,
        params: RiskParams | None = None,
        image_center: tuple[float, float] = (160.0, 160.0),
        image_area: float = 102400.0,
    ):
        self.params = params or RiskParams()
        self.cx0, self.cy0 = image_center
        self.corner_dist = ((image_center[0])**2 + (image_center[1])**2) ** 0.5
        self.image_area = image_area

        # 每个目标的 EMA 历史 & 面积历史
        self._history: dict[int, dict[str, Any]] = {}

    def evaluate(
        self,
        track_id: int,
        area: float,
        confidence: float,
        cx: float,
        cy: float,
    ) -> TargetRisk:
        """评估单个目标的风险。

        Returns:
            TargetRisk with raw_risk, smoothed_risk, state
        """
        hist = self._history.get(track_id, {})
        prev_area = hist.get("last_area", area)
        prev_smoothed = hist.get("smoothed_risk", 0.0)

        # 1. 中心距离评分
        dist = ((cx - self.cx0)**2 + (cy - self.cy0)**2) ** 0.5
        center_score = 1.0 - min(1.0, dist / self.corner_dist)

        # 2. 面积评分（相对全图面积）
        area_score = min(1.0, area / self.image_area * 10.0)

        # 3. 面积增长率
        growth_rate = (area - prev_area) / max(prev_area, 1.0)
        # 映射到 [0, 1]：+50% → 1.0, -50% → 0.0
        growth_score = max(0.0, min(1.0, (growth_rate + 0.5) / 1.0))

        # 4. 置信度
        conf_score = min(1.0, max(0.0, confidence))

        # ---- 原始风险 ----
        raw = (
            self.params.w_center * center_score
            + self.params.w_area * area_score
            + self.params.w_growth * growth_score
            + self.params.w_confidence * conf_score
        )

        # ---- EMA 平滑 ----
        alpha = self.params.ema_alpha
        smoothed = alpha * raw + (1.0 - alpha) * prev_smoothed

        # 保存历史
        self._history[track_id] = {
            "last_area": area,
            "smoothed_risk": smoothed,
        }

        risk = TargetRisk(
            track_id=track_id,
            raw_risk=round(raw, 4),
            smoothed_risk=round(smoothed, 4),
            center_score=round(center_score, 4),
            area_score=round(area_score, 4),
            growth_rate=round(growth_rate, 4),
            confidence=round(conf_score, 4),
            last_area=prev_area,
        )
        risk.update_state()
        return risk

    def get_history(self, track_id: int) -> dict[str, Any] | None:
        return self._history.get(track_id)

    def forget(self, track_id: int) -> None:
        """目标丢失后清理历史。"""
        self._history.pop(track_id, None)

    def prune(self, active_ids: set[int]) -> None:
        """清理不再活跃的目标历史。"""
        stale = set(self._history.keys()) - active_ids
        for tid in stale:
            del self._history[tid]
