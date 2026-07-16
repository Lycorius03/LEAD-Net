"""多教师渐进式蒸馏训练器。

论文: MAID (2025), PDQ-KD (2025)
策略:
    Phase 1 (30% steps): 仅 RT-DETR 蒸馏 → 全局特征对齐
    Phase 2 (40% steps): RT-DETR + YOLO11x 双教师 → 全局+局部
    Phase 3 (30% steps): 仅 YOLO11x + 硬标签 → 精细调优
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiTeacherDistiller:
    """多教师渐进式蒸馏包装器。

    封装学生模型 + 蒸馏损失，处理三阶段训练调度。

    用法::

        distiller = MultiTeacherDistiller(student_model, cfg)
        distiller.set_phase(1)  # RT-DETR only
        for batch in loader:
            outputs = distiller.train_step(images, gt_boxes, gt_labels, rt_detr_out)
    """

    def __init__(self, student: nn.Module, cfg: dict):
        self.student = student
        self.cfg = cfg

        distill_cfg = cfg.get("distill", {})
        self._temperature = distill_cfg.get("temperature", 4.0)
        self._rt_weight = distill_cfg.get("rt_detr_weight", 0.6)
        self._yolo_weight = distill_cfg.get("yolo_weight", 0.4)
        self._phase: int = 0
        self._current_step: int = 0
        self._total_steps: int = 0

    def set_total_steps(self, total: int) -> None:
        """设置训练总步数（用于阶段划分）。"""
        self._total_steps = total

    @property
    def phase(self) -> int:
        """当前蒸馏阶段 0/1/2/3。0=无蒸馏（纯硬标签）。"""
        return self._phase

    def update_phase(self, step: int) -> int:
        """根据当前步数更新蒸馏阶段。

        Phase 1: step < 30% total → RT-DETR only
        Phase 2: 30-70% → RT-DETR + YOLO
        Phase 3: 70-100% → YOLO only + hard labels
        """
        self._current_step = step
        if self._total_steps <= 0:
            self._phase = 0
            return 0

        progress = step / self._total_steps
        if progress < 0.3:
            self._phase = 1
        elif progress < 0.7:
            self._phase = 2
        else:
            self._phase = 3
        return self._phase

    def compute_distill_loss(self, s_cls: torch.Tensor, s_loc: torch.Tensor,
                             teacher_outputs: dict) -> dict[str, torch.Tensor]:
        """根据当前阶段计算蒸馏损失。

        Args:
            s_cls, s_loc: 学生输出
            teacher_outputs: {"rt_detr": (cls, loc), "yolo": (cls, loc)} 或其中一部分

        Returns:
            dict with distill_loss and per-teacher breakdown
        """
        from lead_net.distill.distill_loss import DistillationLoss, KDLoss

        losses = {"distill_loss": torch.tensor(0.0, device=s_cls.device)}

        t_rt = teacher_outputs.get("rt_detr")
        t_yolo = teacher_outputs.get("yolo")

        if self._phase == 1 and t_rt is not None:
            # RT-DETR only
            dl = DistillationLoss(
                temperature=self._temperature, kld_weight=0.6, l2_weight=0.4, feature_weight=0.0,
            )
            r = dl(s_cls, s_loc, t_rt[0], t_rt[1])
            losses["distill_loss"] = r["distill_loss"]
            losses["rt_kld"] = r["kld_loss"]
            losses["rt_l2"] = r["l2_loss"]

        elif self._phase == 2 and t_rt is not None and t_yolo is not None:
            # Both teachers
            dl_rt = DistillationLoss(
                temperature=self._temperature, kld_weight=0.6, l2_weight=0.4, feature_weight=0.0,
            )
            r_rt = dl_rt(s_cls, s_loc, t_rt[0], t_rt[1])
            dl_yolo = DistillationLoss(
                temperature=self._temperature, kld_weight=0.6, l2_weight=0.4, feature_weight=0.0,
            )
            r_yolo = dl_yolo(s_cls, s_loc, t_yolo[0], t_yolo[1])
            losses["distill_loss"] = (self._rt_weight * r_rt["distill_loss"]
                                      + self._yolo_weight * r_yolo["distill_loss"])
            losses["rt_kld"] = r_rt["kld_loss"]
            losses["rt_l2"] = r_rt["l2_loss"]
            losses["yolo_kld"] = r_yolo["kld_loss"]
            losses["yolo_l2"] = r_yolo["l2_loss"]

        elif self._phase == 3 and t_yolo is not None:
            # YOLO only (fine-tuning)
            dl = DistillationLoss(
                temperature=self._temperature, kld_weight=0.3, l2_weight=0.7, feature_weight=0.0,
            )
            r = dl(s_cls, s_loc, t_yolo[0], t_yolo[1])
            losses["distill_loss"] = r["distill_loss"]
            losses["yolo_kld"] = r["kld_loss"]
            losses["yolo_l2"] = r["l2_loss"]

        return losses

    def mock_teacher_output(self, device: torch.device,
                            num_anchors: int = 2475,
                            num_classes: int = 8) -> dict:
        """生成模拟教师输出（冒烟测试用）。

        Returns:
            {"rt_detr": (cls_logits, loc), "yolo": (cls_logits, loc)}
        """
        B = 2  # batch size for smoke test
        rt_cls = torch.randn(B, num_anchors, num_classes, device=device)
        rt_loc = torch.randn(B, num_anchors, 4, device=device)
        yolo_cls = torch.randn(B, num_anchors, num_classes, device=device)
        yolo_loc = torch.randn(B, num_anchors, 4, device=device)
        return {"rt_detr": (rt_cls, rt_loc), "yolo": (yolo_cls, yolo_loc)}
