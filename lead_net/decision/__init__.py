"""避障决策模块 —— 三层决策流程 + DL/CV 融合。

对应 PLAN §M9，服务 RQ6（通用前景感知与混合避障策略）。

提供：
    - DecisionEngine：三层决策引擎（置信度→ROI→面积代理）
    - DecisionResult：决策输出数据结构
    - ROIFilter / PriorityCalculator / FusionStrategy：各层独立组件

参考：
    - R8 (ADOS, IEEE IV 2024): class-agnostic 避障感知
    - R14 (J-MOD²): bbox 与深度联合学习
    - R16 (DECADE, 2024): 纯 bbox 几何推理碰撞避免
"""

from .decision_engine import DecisionEngine, DecisionResult, DecisionParams
from .roi_filter import ROIFilter, ROIParams
from .priority import PriorityCalculator, PriorityParams
from .fusion import FusionStrategy, FusionParams

__all__ = [
    "DecisionEngine",
    "DecisionResult",
    "DecisionParams",
    "ROIFilter",
    "ROIParams",
    "PriorityCalculator",
    "PriorityParams",
    "FusionStrategy",
    "FusionParams",
]
