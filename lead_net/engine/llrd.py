"""LLRD（Layer-wise Learning Rate Decay）—— 五级分层学习率 + 阶段过渡。

五级划分：
    Head (SSD-Lite)     → 1.0e-3
    LCA (Attention)     → 8.0e-4
    Backbone 最后几层   → 3e-4
    Backbone 中间层     → 1e-4
    Backbone 前几层     → 3e-5

每个 LR 组自动拆分为 weight_decay / no_weight_decay 两组，
BN 和 bias 参数不参与 weight decay。

阶段过渡：
    Stage 1 → Stage 2: unfreeze_backbone() 恢复 requires_grad，
    然后重新调用 build_llrd_param_groups(freeze_backbone=False) 重建优化器。
"""

from __future__ import annotations

import torch.nn as nn


_DEFAULT_LR_CONFIG = {
    "head": 1.0e-3,
    "lca": 8.0e-4,
    "backbone_last": 3e-4,
    "backbone_middle": 1e-4,
    "backbone_first": 3e-5,
}


def _is_weight(p: nn.Parameter) -> bool:
    """BN 权重和 bias 都是 1 维参数，不应参与 weight decay。"""
    return p.ndim >= 2


def unfreeze_backbone(model: nn.Module) -> None:
    """恢复 Backbone 所有参数的 requires_grad=True（Stage 1 → Stage 2 过渡时调用）。

    只操作 backbone 内部的参数，确保 BatchNorm 层也被恢复。
    """
    backbone = model.backbone
    for param in backbone.parameters():
        param.requires_grad = True
    # 确保 BN 层处于训练模式（fine-tuning 时需要更新 running stats）
    for m in backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()


def freeze_backbone(model: nn.Module) -> None:
    """冻结 Backbone 所有参数（Stage 1 开始时调用）。"""
    backbone = model.backbone
    for param in backbone.parameters():
        param.requires_grad = False
    # BN 层设为 eval 模式，保留预训练统计量
    for m in backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def build_llrd_param_groups(
    model: nn.Module,
    cfg: dict | None = None,
    freeze_backbone: bool = False,
) -> list[dict]:
    """构建 LLRD 参数组列表，可直接传给 torch.optim.SGD。

    Args:
        model: LEAD-Net 模型
        cfg: 配置（读 learning_rate / optimizer 段）
        freeze_backbone: 阶段一时冻结 Backbone

    Returns:
        param_groups 列表，每组含 params / lr / weight_decay / name / initial_lr
    """
    lr_cfg = dict(_DEFAULT_LR_CONFIG)
    if cfg:
        lr_cfg.update(cfg.get("learning_rate", {}))
    wd = cfg.get("optimizer", {}).get("weight_decay", 5e-4) if cfg else 5e-4

    backbone = model.backbone
    features = backbone.features
    n = len(features)
    t1 = max(1, n // 3)
    t2 = max(1, 2 * n // 3)

    # 收集各组的 (params_with_wd, params_without_wd)
    groups_raw: list[tuple[str, float, list, list]] = []  # [(name, lr, wd_list, nowd_list)]

    # Head
    head_w, head_nw = _split_wd(list(model.head.parameters()))
    groups_raw.append(("head", lr_cfg["head"], head_w, head_nw))

    # LCA
    lca_mod = getattr(backbone, "lca", None)
    if lca_mod is not None:
        lca_w, lca_nw = _split_wd(list(lca_mod.parameters()))
        groups_raw.append(("lca", lr_cfg["lca"], lca_w, lca_nw))

    # Backbone 三等分
    bb_first, bb_mid, bb_last = [], [], []
    for i, child in enumerate(features.children()):
        plist = list(child.parameters())
        if i < t1:
            bb_first.extend(plist)
        elif i < t2:
            bb_mid.extend(plist)
        else:
            bb_last.extend(plist)
    for attr in ("proj_s16", "proj_s32", "extra"):
        mod = getattr(backbone, attr, None)
        if mod is not None:
            bb_last.extend(list(mod.parameters()))

    if freeze_backbone:
        # 冻结 Backbone：设置 requires_grad=False，不创建 LR 组
        all_bb = bb_first + bb_mid + bb_last
        for p in all_bb:
            p.requires_grad = False
    else:
        # 确保 Backbone 参数可训练
        all_bb = bb_first + bb_mid + bb_last
        for p in all_bb:
            p.requires_grad = True

        for name, lr_val, params in [
            ("backbone_last", lr_cfg["backbone_last"], bb_last),
            ("backbone_middle", lr_cfg["backbone_middle"], bb_mid),
            ("backbone_first", lr_cfg["backbone_first"], bb_first),
        ]:
            w_list, nw_list = _split_wd(params)
            groups_raw.append((name, lr_val, w_list, nw_list))

    # 输出优化器格式
    result = []
    for name, lr_val, w_params, nw_params in groups_raw:
        if w_params:
            result.append({"params": w_params, "lr": lr_val, "weight_decay": wd,
                           "name": f"{name}_wd", "initial_lr": lr_val})
        if nw_params:
            result.append({"params": nw_params, "lr": lr_val, "weight_decay": 0.0,
                           "name": f"{name}_nowd", "initial_lr": lr_val})

    return result


def _split_wd(params: list[nn.Parameter]) -> tuple[list, list]:
    """分离需要/不需要 weight decay 的参数。"""
    w_list, nw_list = [], []
    for p in params:
        if not p.requires_grad:
            continue
        if _is_weight(p):
            w_list.append(p)
        else:
            nw_list.append(p)
    return w_list, nw_list
