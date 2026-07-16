"""知识蒸馏模块 — 多教师渐进式蒸馏（RT-DETR-L + YOLO11x → LEAD-Net）。

论文依据:
    - MAID: Multi-Teacher Adaptive Instance Distillation (2025)
    - PDQ-KD: Progressive Object Query KD for DETR (2025)
    - LADNet: RT-DETR→MobileNet KD verified (ScienceDirect 2025)
    - DetKDS: Knowledge Distillation Search (ICML 2024)
"""
from lead_net.distill.distill_loss import DistillationLoss, KDLoss
from lead_net.distill.multi_teacher_kd import MultiTeacherDistiller

__all__ = ["DistillationLoss", "KDLoss", "MultiTeacherDistiller"]
