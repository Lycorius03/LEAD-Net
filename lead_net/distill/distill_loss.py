"""蒸馏损失函数 — KLD + L2 + Feature Distillation。

论文: PDQ-KD (2025), CWD+BCKD (2024)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """KL 散度蒸馏损失（logit-level）。

    Args:
        temperature: 软化温度 T，默认 4.0。
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """计算 KL 散度损失。

        Args:
            student_logits: [B, N, C] 学生分类 logits
            teacher_logits: [B, N, C] 教师分类 logits（需对齐 anchor 维度）

        Returns:
            KL loss (scalar)
        """
        # 确保维度对齐
        if student_logits.shape != teacher_logits.shape:
            # 简单截断/填充到相同 N
            min_n = min(student_logits.size(1), teacher_logits.size(1))
            student_logits = student_logits[:, :min_n, :]
            teacher_logits = teacher_logits[:, :min_n, :]

        s_log_prob = F.log_softmax(student_logits / self.T, dim=-1)
        t_prob = F.softmax(teacher_logits / self.T, dim=-1)
        kld = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (self.T ** 2)
        return kld


class DistillationLoss(nn.Module):
    """综合蒸馏损失 — KLD (logit) + L2 (regression) + Feature (可选)。

    多教师权重: RT-DETR=0.6, YOLO=0.4

    Args:
        temperature: KL 温度
        kld_weight: KL 散度权重
        l2_weight: L2 回归损失权重
        feature_weight: 特征蒸馏权重（0=禁用）
    """

    def __init__(self, temperature: float = 4.0, kld_weight: float = 0.5,
                 l2_weight: float = 0.3, feature_weight: float = 0.2):
        super().__init__()
        self.kld = KDLoss(temperature)
        self.kld_weight = kld_weight
        self.l2_weight = l2_weight
        self.feature_weight = feature_weight

    def forward(self, s_cls: torch.Tensor, s_loc: torch.Tensor,
                t_cls: torch.Tensor, t_loc: torch.Tensor,
                s_feat: torch.Tensor | None = None,
                t_feat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """计算综合蒸馏损失。

        Returns:
            dict with keys: distill_loss, kld_loss, l2_loss, feat_loss
        """
        losses = {}

        # KL 散度（分类）
        kld_loss = self.kld(s_cls, t_cls)
        losses["kld_loss"] = kld_loss

        # L2（回归）
        if s_loc.shape != t_loc.shape:
            min_n = min(s_loc.size(1), t_loc.size(1))
            s_loc = s_loc[:, :min_n, :]
            t_loc = t_loc[:, :min_n, :]
        l2_loss = F.mse_loss(s_loc, t_loc)
        losses["l2_loss"] = l2_loss

        # 特征蒸馏（可选）
        feat_loss = torch.tensor(0.0, device=s_cls.device)
        if s_feat is not None and t_feat is not None and self.feature_weight > 0:
            feat_loss = F.mse_loss(s_feat, t_feat)
        losses["feat_loss"] = feat_loss

        total = (self.kld_weight * kld_loss + self.l2_weight * l2_loss
                 + self.feature_weight * feat_loss)
        losses["distill_loss"] = total

        return losses


class MultiTeacherLoss(nn.Module):
    """多教师蒸馏损失 — 加权合并 RT-DETR + YOLO 教师。

    Args:
        rt_detr_weight: RT-DETR 教师权重（默认 0.6）
        yolo_weight: YOLO 教师权重（默认 0.4）
        temperature: KL 温度
    """

    def __init__(self, rt_detr_weight: float = 0.6, yolo_weight: float = 0.4,
                 temperature: float = 4.0):
        super().__init__()
        self.rt_detr_loss = DistillationLoss(temperature=temperature)
        self.yolo_loss = DistillationLoss(temperature=temperature)
        self.w_rt = rt_detr_weight
        self.w_yolo = yolo_weight

    def forward(self, s_cls, s_loc, s_feat,
                rt_cls, rt_loc, rt_feat,
                yolo_cls, yolo_loc, yolo_feat) -> dict[str, torch.Tensor]:
        """计算双教师综合蒸馏损失。"""
        rt = self.rt_detr_loss(s_cls, s_loc, rt_cls, rt_loc, s_feat, rt_feat)
        yolo = self.yolo_loss(s_cls, s_loc, yolo_cls, yolo_loc, s_feat, yolo_feat)

        return {
            "distill_loss": self.w_rt * rt["distill_loss"] + self.w_yolo * yolo["distill_loss"],
            "rt_detr_kld": rt["kld_loss"],
            "rt_detr_l2": rt["l2_loss"],
            "yolo_kld": yolo["kld_loss"],
            "yolo_l2": yolo["l2_loss"],
        }
