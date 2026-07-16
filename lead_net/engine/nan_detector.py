"""NaN/Inf 诊断工具 —— Fairseq 风格的 hook-based 检测器。

用法:
    from lead_net.engine.nan_detector import NanDetector, grad_stats

    # 方法 1: 定位 NaN 首次出现的精确层
    with NanDetector(model) as nd:
        loss.backward()
        if nd.found_nan:
            print(f"NaN at: {nd.first_nan_module}")

    # 方法 2: 快速统计全部层的梯度状态
    stats = grad_stats(model)
    for name, s in stats.items():
        if s['has_nan']:
            print(f"NaN in: {name}")

参考:
    Fairseq NanDetector (https://github.com/facebookresearch/fairseq)
    PyTorch autograd anomaly detection
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any


class NanDetector:
    """注册所有子模块的 forward/backward hooks，在首次 NaN/Inf 时报告。

    使用 context manager 自动管理 hook 生命周期::

        with NanDetector(model) as nd:
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
        if nd.found_nan:
            print(f"First NaN at: {nd.first_nan_module}")
    """

    def __init__(self, model: nn.Module, forward: bool = True, backward: bool = True):
        self.model = model
        self.fhooks: list[torch.utils.hooks.RemovableHandle] = []
        self.bhooks: list[torch.utils.hooks.RemovableHandle] = []
        self.forward_enabled = forward
        self.backward_enabled = backward
        self._found_forward = False
        self._found_backward = False
        self.first_nan_module: str | None = None
        self.first_nan_direction: str | None = None  # "forward" or "backward"
        self.found_nan = False

    def __enter__(self):
        for name, mod in self.model.named_modules():
            # 跳过没有参数的容器模块（减少 hook 开销）
            if self._is_leaf_module(mod):
                mod.__nan_detector_name = name  # type: ignore[attr-defined]
                if self.forward_enabled:
                    h = mod.register_forward_hook(self._fhook)
                    self.fhooks.append(h)
                if self.backward_enabled:
                    h = mod.register_full_backward_hook(self._bhook)
                    self.bhooks.append(h)
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """手动移除所有 hooks。"""
        for h in self.fhooks:
            h.remove()
        for h in self.bhooks:
            h.remove()
        self.fhooks.clear()
        self.bhooks.clear()

    def _detect(self, tensor: torch.Tensor, name: str, direction: str) -> str | None:
        """检测单个 tensor 中的 NaN/Inf。"""
        if not torch.is_floating_point(tensor):
            return None
        # 用小批量检查（避免逐元素全量扫描）
        if tensor.numel() == 0:
            return None
        # 采样检查：取第一个和最后一个 batch 元素的 stats
        with torch.no_grad():
            if torch.isnan(tensor).any():
                return f"[{direction}] NaN in output of '{name}'"
            elif torch.isinf(tensor).any():
                return f"[{direction}] Inf in output of '{name}'"
        return None

    def _fhook(self, module: nn.Module, args: tuple, output: Any):
        if self._found_forward:
            return
        name = getattr(module, "__nan_detector_name", "unknown")
        # 输入也可能包含 NaN
        for i, inp in enumerate(args):
            if isinstance(inp, torch.Tensor):
                err = self._detect(inp, f"{name}.input[{i}]", "forward")
                if err and not self._found_forward:
                    self._found_forward = True
                    self.first_nan_module = err
                    self.first_nan_direction = "forward"
                    self.found_nan = True
        # 输出检测
        if not self._found_forward:
            if isinstance(output, torch.Tensor):
                err = self._detect(output, name, "forward")
                if err:
                    self._found_forward = True
                    self.first_nan_module = err
                    self.first_nan_direction = "forward"
                    self.found_nan = True

    def _bhook(self, module: nn.Module, grad_input: tuple, grad_output: tuple):
        if self._found_backward:
            return
        name = getattr(module, "__nan_detector_name", "unknown")
        for i, g in enumerate(grad_output):
            if g is not None:
                err = self._detect(g, f"{name}.grad[{i}]", "backward")
                if err:
                    self._found_backward = True
                    self.first_nan_module = err
                    self.first_nan_direction = "backward"
                    self.found_nan = True
                    break

    @staticmethod
    def _is_leaf_module(mod: nn.Module) -> bool:
        """判断是否为『叶子』模块（有参数或为基本运算层）。"""
        # 有参数的模块
        if len(list(mod.parameters(recurse=False))) > 0:
            return True
        # 基本运算模块（如激活函数、池化等，可能是 NaN 来源）
        leaf_types = (
            nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh,
            nn.Softmax, nn.LogSoftmax,
            nn.Dropout, nn.Dropout2d,
            nn.BatchNorm2d, nn.LayerNorm,
            nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
        )
        if isinstance(mod, leaf_types):
            return True
        return False


def grad_stats(model: nn.Module) -> dict[str, dict]:
    """快速统计模型每层的梯度状态（零开销，不注册 hooks）。

    在每个 optimizer.step() 之后调用。

    Returns:
        dict: {param_name: {"min": float, "max": float, "mean": float,
                             "std": float, "norm": float, "has_nan": bool, "has_inf": bool}}
        无梯度的参数不包含在结果中。
    """
    result = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.detach()
        g_flat = g.flatten()
        g_norm = g.norm().item()
        # 快速 NaN/Inf 检测
        is_nan = torch.isnan(g).any().item()
        is_inf = torch.isinf(g).any().item()
        if g_flat.numel() <= 1:
            result[name] = {
                "value": g_flat.item(),
                "norm": g_norm,
                "has_nan": is_nan,
                "has_inf": is_inf,
            }
        else:
            result[name] = {
                "min": g_flat.min().item(),
                "max": g_flat.max().item(),
                "mean": g_flat.mean().item(),
                "std": g_flat.std().item(),
                "norm": g_norm,
                "has_nan": is_nan,
                "has_inf": is_inf,
            }
    return result


def summarize_grads(stats: dict[str, dict]) -> str:
    """格式化为单行摘要，适合训练日志输出。

    Returns:
        描述梯度状态的字符串，如 "OK" 或 "NaN in [head.loc_head.0.weight, ...]"
    """
    nan_params = [n for n, s in stats.items() if s.get("has_nan")]
    inf_params = [n for n, s in stats.items() if s.get("has_inf") and not s.get("has_nan")]
    large_params = [(n, s["norm"]) for n, s in stats.items()
                    if s.get("norm", 0) > 100 and not s.get("has_nan")]
    if nan_params:
        return f"NaN in [{', '.join(nan_params[:3])}{'...' if len(nan_params) > 3 else ''}]"
    if inf_params:
        return f"Inf in [{', '.join(inf_params[:3])}{'...' if len(inf_params) > 3 else ''}]"
    if large_params:
        top3 = sorted(large_params, key=lambda x: -x[1])[:3]
        return f"Large grad: {', '.join(f'{n}={v:.1f}' for n, v in top3)}"
    return "OK"
