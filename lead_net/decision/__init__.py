"""避障决策模块 —— 风险评估器架构。

对应 PLAN §M9，服务 RQ6。

提供：
    - DecisionEngine：五层风险评估引擎（置信度→ROI→分组面积→Kalman→风险评分）
    - DecisionResult：(x, y, a) + risk_score + state + UART 格式化
    - RiskCalculator：统一多因素风险评分（EMA 平滑 + 增长率）
    - PriorityCalculator：双组面积阈值紧急度判定
    - ROIFilter / FusionStrategy：各层独立组件

参考：
    - 最终避障决策方案 + 用户优化建议（2026-07-15）
"""

from .decision_engine import DecisionEngine, DecisionResult, DecisionParams
from .roi_filter import ROIFilter, ROIParams
from .priority import PriorityCalculator, PriorityParams
from .risk import RiskCalculator, RiskParams, TargetRisk
from .fusion import FusionStrategy, FusionParams

__all__ = [
    "DecisionEngine", "DecisionResult", "DecisionParams",
    "ROIFilter", "ROIParams",
    "PriorityCalculator", "PriorityParams",
    "RiskCalculator", "RiskParams", "TargetRisk",
    "FusionStrategy", "FusionParams",
]
