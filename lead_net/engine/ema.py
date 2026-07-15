"""EMA（指数移动平均）模型包装器。

训练时维护一份参数的指数移动平均副本，验证和最终导出使用 EMA 版本。
工业标准（YOLOv5 起默认开启），预期 mAP +0.3%~1%，代价极低。

用法:
    ema = ModelEMA(model, decay=0.9998)
    for epoch in range(epochs):
        train_one_epoch(...)
        ema.update(model)                    # 每个 iteration 后调用
        ema.apply()                          # 验证前：切换到 EMA 参数
        validate(...)
        ema.restore()                        # 验证后：切回训练参数
    ema.apply_permanently()                  # 训练结束：永久切换到 EMA
    torch.save(model.state_dict(), "best.pt")
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """模型参数指数移动平均。

    Args:
        model: 训练中的模型
        decay: EMA 衰减系数（默认 0.9998，每步更新 0.02%）
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.model = model
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        self._init_shadow()

    def _init_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module | None = None) -> None:
        """每个 iteration 后调用一次。可传入 model 或使用构造函数中的 model。"""
        src = model if model is not None else self.model
        for name, param in src.named_parameters():
            if param.requires_grad:
                self._shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self) -> None:
        """将 EMA 参数应用到模型上（验证/导出前调用）。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._backup[name] = param.data.clone()
                param.data.copy_(self._shadow[name])

    def restore(self) -> None:
        """恢复训练参数（验证后调用）。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup.pop(name))

    def apply_permanently(self) -> None:
        """训练结束后永久切换到 EMA 版本。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self._shadow[name])
        self._shadow.clear()
        self._backup.clear()
