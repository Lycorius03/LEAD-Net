"""伪量化节点 — INT8 FakeQuantize for QAT。

论文: Jacob et al., CVPR 2018; α-QAT, IEEE Aug 2024

核心思路:
    训练时模拟 INT8 量化的取整误差，前向使用量化值，反向使用 STE (Straight-Through Estimator)。
    α-QAT 改进: 使用仿射组合替代纯 STE，减少梯度偏差。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FakeQuantize(nn.Module):
    """INT8 伪量化模块（per-tensor symmetric）。

    Args:
        qmin: 量化最小值，默认 -128 (INT8)
        qmax: 量化最大值，默认 127 (INT8)
        use_alpha: 是否启用 α-QAT 改进
        alpha: α 参数（仅在 use_alpha=True 时使用）
    """

    def __init__(self, qmin: int = -128, qmax: int = 127,
                 use_alpha: bool = False, alpha: float = 0.5):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.use_alpha = use_alpha
        self.alpha = alpha
        self.scale: torch.Tensor | None = None  # 在 forward 中计算

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """伪量化前向传播。

        quant = round(clamp(x / scale, qmin, qmax))
        dequant = quant * scale

        α-QAT: output = α * dequant + (1-α) * x（训练早期更多全精度）
        """
        if self.scale is None or not self.training:
            # 动态计算 scale
            self.scale = x.abs().max() / self.qmax

        scale = self.scale.to(x.device)

        # 量化-反量化
        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, self.qmin, self.qmax)
        x_rounded = torch.round(x_clamped)
        x_dequant = x_rounded * scale

        if self.training and self.use_alpha:
            # α-QAT: 仿射组合，STE 仅用于量化部分
            output = self.alpha * x_dequant + (1.0 - self.alpha) * x
            # 梯度通过 STE 传播
            output = x + (output - x).detach()
        else:
            # 标准 STE
            output = x + (x_dequant - x).detach()

        return output


def prepare_qat(model: nn.Module, use_alpha: bool = True) -> nn.Module:
    """为模型插入伪量化节点（简化版：仅对 Conv2d 权重量化）。

    在 Conv2d 权重后插入 FakeQuantize，模拟 INT8 推理。

    Args:
        model: 待量化的模型
        use_alpha: 是否使用 α-QAT

    Returns:
        插入了 FakeQuantize 的模型（就地修改）
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 注册伪量化 hook
            fq = FakeQuantize(use_alpha=use_alpha)
            _register_weight_quant_hook(module, fq)
    return model


def _register_weight_quant_hook(conv: nn.Conv2d, fq: FakeQuantize) -> None:
    """注册权重伪量化 forward hook。"""
    def hook(module, input, output):
        # 对权重进行伪量化后再做卷积
        fake_weight = fq(module.weight)
        if module.bias is not None:
            return nn.functional.conv2d(
                input[0], fake_weight, module.bias,
                module.stride, module.padding, module.dilation, module.groups,
            )
        else:
            return nn.functional.conv2d(
                input[0], fake_weight, None,
                module.stride, module.padding, module.dilation, module.groups,
            )

    conv.register_forward_hook(hook)
