"""engine 子包：Trainer / Evaluator / MetricsCollector / EvalAnalysis。"""

from .trainer import Trainer
from .evaluator import Evaluator
from .metrics import MetricsCollector
from .eval_analysis import analyze_coco_eval, run_analysis_from_predictions

__all__ = [
    "Trainer", "Evaluator", "MetricsCollector",
    "analyze_coco_eval", "run_analysis_from_predictions",
]
