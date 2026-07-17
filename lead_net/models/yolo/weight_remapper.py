"""weight_remapper.py — YOLO11 插入 LCA 后预训练权重键名重映射。

职责（单一）：
    LCA 插入导致 head 层索引偏移，ultralytics .load() 按键名匹配会漏掉 head 权重。
    本模块把预训练 csd 的键名按偏移量重映射，使 head 权重正确迁移。

机制：
    baseline 层 0-16 与 lca 层 0-16 一致（backbone + 部分 head）
    baseline 层 17-23 对应 lca 层 18-24（LCA 在 17 插入）
    重映射：csd["model.17.xxx"] → "model.18.xxx", ..., csd["model.23.xxx"] → "model.24.xxx"
    Detect 层的 from 索引 [16,19,22]→[16,20,23] 不影响权重迁移（只是 forward 时的来源）

不负责：
    - LCA 模块本体
    - parse_model
    - 训练流程
"""
from __future__ import annotations

from collections import OrderedDict

import torch


def remap_csd_for_lca_insertion(
    csd: dict[str, torch.Tensor],
    lca_insert_index: int = 17,
    num_baseline_layers: int = 24,  # baseline 0-23
) -> OrderedDict[str, torch.Tensor]:
    """重映射预训练 csd 键名，补偿 LCA 插入导致的索引偏移。

    Args:
        csd: 预训练模型的 state_dict（键名 "model.{i}.{...}"）
        lca_insert_index: LCA 插入的层索引（baseline 的该索引及之后 → lca 的 +1）
        num_baseline_layers: baseline 模型总层数

    Returns:
        重映射后的 csd，键名与 lca 模型对齐
    """
    remapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    offset = 1  # LCA 插入 1 层

    for key, val in csd.items():
        # 键名格式 "model.{i}.{sub}" 或 "model.{i}.0.{sub}" 等
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "model":
            try:
                layer_idx = int(parts[1])
            except ValueError:
                remapped[key] = val
                continue

            # backbone 部分（< lca_insert_index）：直接保留
            if layer_idx < lca_insert_index:
                remapped[key] = val
            # LCA 插入位置及之后（>= lca_insert_index）：+offset
            elif layer_idx >= lca_insert_index and layer_idx < num_baseline_layers:
                new_idx = layer_idx + offset
                new_key = f"model.{new_idx}." + ".".join(parts[2:])
                # 只在目标模型有对应形状时迁移（LCA 层 model.17 无对应预训练权重，跳过）
                remapped[new_key] = val
            else:
                # 超出 baseline 范围的（如 num_classes 头等额外项）
                remapped[key] = val
        else:
            # 非 "model.{i}" 格式的键（如 trainer 相关）直接保留
            remapped[key] = val

    return remapped


def load_pretrained_with_remapping(
    model_yaml: str,
    pretrained_pt: str,
    lca_insert_index: int = 17,
    verbose: bool = True,
):
    """构建模型 + 重映射权重 + 加载。

    Args:
        model_yaml: YAML 路径（含 LCA 插入）
        pretrained_pt: 预训练权重路径（如 yolo11n.pt）
        lca_insert_index: LCA 插入位置
        verbose: 打印迁移进度
    Returns:
        ultralytics YOLO 模型（权重已正确迁移）
    """
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    # 构建模型（从 YAML，不加载预训练）
    model = YOLO(model_yaml)

    # 加载预训练权重（torch 2.6+ 默认 weights_only=True，需显式 False 加载 ultralytics checkpoint）
    weights = torch.load(pretrained_pt, map_location="cpu", weights_only=False)
    csd = weights["model"].float().state_dict() if isinstance(weights, dict) and "model" in weights else weights.float().state_dict()

    # 重映射
    remapped_csd = remap_csd_for_lca_insertion(csd, lca_insert_index)

    # intersect + load
    from ultralytics.utils.torch_utils import intersect_dicts
    model_state = model.model.state_dict()
    updated_csd = intersect_dicts(remapped_csd, model_state)
    # 形状匹配过滤（intersect_dicts 已做，但重映射后某些层形状可能变）
    final_csd = {k: v for k, v in updated_csd.items() if k in model_state and model_state[k].shape == v.shape}
    model.model.load_state_dict(final_csd, strict=False)

    if verbose:
        LOGGER.info(f"Transferred {len(final_csd)}/{len(model_state)} items from pretrained weights (with LCA remapping)")

    # 关键：标记 ckpt 非空，使 model.train() 复用当前内存权重（2026-07-18 修复）。
    # ultralytics engine/model.py:808 只在 self.ckpt 为 truthy 时才把 self.model
    # 交给 trainer；YOLO(yaml) 构建的模型 ckpt 为空 → trainer 用 yaml 冷启动
    # 随机初始化，重映射权重被整体丢弃（云端 smoke 实测 lca mAP≈0.001）。
    # 不携带 epoch/optimizer，避免误触发 resume 逻辑（engine/model.py:797）。
    model.ckpt = {"model": model.model}

    return model


if __name__ == "__main__":
    # 自测
    m = load_pretrained_with_remapping(
        "lead_net/models/yolo/yamls/yolo11n_lca_neck_r16.yaml",
        "yolo11n.pt",
        lca_insert_index=17,
    )
    print(f"params: {sum(p.numel() for p in m.model.parameters())/1e6:.4f} M")
