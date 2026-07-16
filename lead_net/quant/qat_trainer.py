"""QAT 训练包装器 — 量化感知训练管理。

用法::

    wrapper = QATWrapper(model)
    wrapper.prepare_qat()     # 插入伪量化节点
    # ... 训练几个 epoch ...
    wrapper.convert_to_int8() # 转为实际 INT8 权重（用于导出）
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QATWrapper:
    """QAT 训练包装器。

    Args:
        model: 待量化的 FP32 模型
        use_alpha_qat: 是否使用 α-QAT（仿射 STE 改进）
    """

    def __init__(self, model: nn.Module, use_alpha_qat: bool = True):
        self.model = model
        self.use_alpha_qat = use_alpha_qat
        self._qat_prepared = False

    def prepare_qat(self) -> nn.Module:
        """插入伪量化节点，准备 QAT 训练。"""
        from lead_net.quant.fake_quant import prepare_qat
        prepare_qat(self.model, use_alpha=self.use_alpha_qat)
        self._qat_prepared = True
        return self.model

    def convert_to_int8(self) -> dict[str, torch.Tensor]:
        """将 QAT 训练后的浮点权重转换为 INT8。

        Returns:
            {"weight_int8": ..., "scales": ..., "zero_points": ...}
            每个 key 对应一个 state_dict-like 的参数字典。
        """
        int8_weights = {}
        scales = {}

        for name, param in self.model.named_parameters():
            if "weight" in name and param.ndim >= 2:
                w = param.data
                w_max = w.abs().max()
                scale = w_max / 127.0
                w_int8 = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
                int8_weights[name] = w_int8
                scales[name] = scale.item()

        return {
            "weight_int8": int8_weights,
            "scales": scales,
        }

    @property
    def is_prepared(self) -> bool:
        return self._qat_prepared

    def export_calibration_stats(self) -> dict[str, dict]:
        """导出校准统计（用于 TFLite 转换）。"""
        stats = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "scale"):
                stats[name] = {"scale": module.scale}
        return stats
