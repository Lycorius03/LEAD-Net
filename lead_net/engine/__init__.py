"""engine 子包：Trainer / Evaluator / MetricsCollector / EvalAnalysis / Scheduler / Checkpoint。"""

from .trainer import Trainer
from .evaluator import Evaluator
from .metrics import MetricsCollector
from .eval_analysis import analyze_coco_eval, run_analysis_from_predictions
from .scheduler import build_scheduler, build_scheduler_from_total_iters
from .checkpoint import CheckpointManager
from .llrd import build_llrd_param_groups, freeze_backbone, unfreeze_backbone

__all__ = [
    "Trainer", "Evaluator", "MetricsCollector",
    "analyze_coco_eval", "run_analysis_from_predictions",
    "build_scheduler", "build_scheduler_from_total_iters",
    "CheckpointManager",
    "build_llrd_param_groups", "freeze_backbone", "unfreeze_backbone",
]
