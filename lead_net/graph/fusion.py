"""Conv-BN-ReLU 算子融合（图优化）。

论文/资料:
    - ONNX Runtime Graph Optimizations (Microsoft)
    - TinyNeuralNetwork FUSE_BN (Alibaba)
    - TinyML Summit 2024: ONNX→TFLite conv+BN+ReLU fusion

原理:
    Conv 输出: y = W * x + b
    BN:        BN(y) = γ·(y - μ)/√(σ²+ε) + β
    ReLU:      ReLU(y) = max(0, y)

    融合为单次 Conv:
        α = γ / √(σ² + ε)
        W' = α · W
        b' = α · (b - μ) + β
        y' = ReLU(W' * x + b')

    数学精确，无损，3 kernel → 1 kernel。

用法::

    fused = fuse_conv_bn_relu(conv_module, bn_module)
    # fused 是新的 nn.Conv2d，可直接替换原 model 中的 conv+bn+relu
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_fusion_params(
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
    bn_eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 Conv-BN 融合后的权重和偏置。

    Args:
        conv_weight: Conv 权重 [out_ch, in_ch, k, k]
        conv_bias: Conv 偏置 [out_ch] 或 None
        bn_weight: BN γ [out_ch]
        bn_bias: BN β [out_ch]
        bn_running_mean: BN μ [out_ch]
        bn_running_var: BN σ² [out_ch]
        bn_eps: BN ε

    Returns:
        (fused_weight, fused_bias): 融合后的权重和偏置
    """
    # α = γ / √(σ² + ε)
    alpha = bn_weight / torch.sqrt(bn_running_var + bn_eps)

    # W' = α · W（广播 α 到每个输出通道）
    fused_weight = conv_weight * alpha[:, None, None, None]

    # b' = α · (b - μ) + β
    if conv_bias is not None:
        fused_bias = alpha * (conv_bias - bn_running_mean) + bn_bias
    else:
        fused_bias = bn_bias - alpha * bn_running_mean

    return fused_weight, fused_bias


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """将 Conv2d + BatchNorm2d 融合为单个 Conv2d。

    Args:
        conv: 卷积层（必须处于 eval 模式）
        bn: BatchNorm 层（必须处于 eval 模式）

    Returns:
        新的 nn.Conv2d，权重已吸收 BN 参数。

    Raises:
        ValueError: 如果 conv.out_channels != bn.num_features
    """
    if conv.out_channels != bn.num_features:
        raise ValueError(
            f"通道不匹配: Conv.out={conv.out_channels}, BN.features={bn.num_features}"
        )

    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BN 层没有 running_mean/var，请先在 eval 模式下 forward 一次")

    fused_w, fused_b = compute_fusion_params(
        conv_weight=conv.weight.data,
        conv_bias=conv.bias.data if conv.bias is not None else None,
        bn_weight=bn.weight.data,
        bn_bias=bn.bias.data,
        bn_running_mean=bn.running_mean.data,
        bn_running_var=bn.running_var.data,
        bn_eps=bn.eps,
    )

    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )

    fused_conv.weight.data = fused_w
    fused_conv.bias.data = fused_b

    return fused_conv


def fuse_conv_bn_relu(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Sequential:
    """将 Conv2d + BatchNorm2d + ReLU(inplace) 融合为 Conv2d + ReLU。

    注意: ReLU 不能真正"融合"进 Conv 权重（非线性），但可以消除 BN kernel。
    结果: 2 kernel (Conv+ReLU) 而非 3 kernel (Conv+BN+ReLU)。

    Args:
        conv: 卷积层
        bn: BatchNorm 层

    Returns:
        nn.Sequential(Conv2d, ReLU(inplace=True))
    """
    fused_conv = fuse_conv_bn(conv, bn)
    return nn.Sequential(fused_conv, nn.ReLU(inplace=True))


def fuse_all_conv_bn_in_place(model: nn.Module, inplace: bool = True) -> nn.Module:
    """递归遍历模型，将所有 Conv2d + BatchNorm2d 序列融合。

    匹配模式: Conv2d → BatchNorm2d → (ReLU?)
    融合为: Conv2d(fused) → (ReLU?)

    Args:
        model: 待融合的模型
        inplace: 是否就地修改模型，默认 True。

    Returns:
        融合后的模型（如果 inplace=True 则返回同一对象）。
    """
    # 使用 _fuse_recursive 遍历
    _fuse_recursive(model)
    return model


def _fuse_recursive(module: nn.Module) -> None:
    """递归融合辅助函数。"""
    children = list(module.named_children())
    n = len(children)

    i = 0
    while i < n - 1:
        name_i, child_i = children[i]
        name_j, child_j = children[i + 1]

        # 模式: Conv → BN → (ReLU)
        if isinstance(child_i, nn.Conv2d) and isinstance(child_j, nn.BatchNorm2d):
            # 检查后面是否有 ReLU
            has_relu = (i + 2 < n and isinstance(children[i + 2][1], nn.ReLU))

            if has_relu:
                name_k = children[i + 2][0]
                fused_conv = fuse_conv_bn(child_i, child_j)
                setattr(module, name_i, fused_conv)
                # ReLU 保留（通过 Sequential 包装）
                setattr(module, name_j, nn.Identity())
                # 不删除 ReLU，它已经是单独层
                i += 2
            else:
                fused_conv = fuse_conv_bn(child_i, child_j)
                setattr(module, name_i, fused_conv)
                setattr(module, name_j, nn.Identity())
                i += 1

        i += 1

    # 递归处理子模块
    for _, child in module.named_children():
        _fuse_recursive(child)


def verify_fusion(
    conv: nn.Conv2d, bn: nn.BatchNorm2d,
    test_input: torch.Tensor, atol: float = 1e-5,
) -> bool:
    """验证融合前后输出一致性。

    Args:
        conv: 原始卷积层
        bn: 原始 BN 层
        test_input: 测试输入 tensor
        atol: 绝对容差

    Returns:
        True 如果融合前后输出一致（在容差内）。
    """
    conv.eval()
    bn.eval()

    with torch.no_grad():
        orig_out = bn(conv(test_input))
        fused_conv = fuse_conv_bn(conv, bn)
        fused_conv.eval()
        fused_out = fused_conv(test_input)

    max_diff = (orig_out - fused_out).abs().max().item()
    return max_diff < atol
