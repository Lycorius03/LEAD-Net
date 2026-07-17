"""lca_adapter.py — 将 LCA 注册到 ultralytics 模块命名空间 + YAML args 对齐。

职责（单一）：
    让 ultralytics 的 parse_model() 能在 YAML 中识别 "LCA" 模块名，
    并把 YAML args 对齐到 LCA 的构造签名 (channels=None, reduction=16)。

机制：
    monkey-patch parse_model，调用前预处理 d['backbone']+d['head'] 的 LCA 行：
    - args == [r]        → [None, r]   （单参数语义为 reduction；channels=None
                                          时构造不建层，首次 forward 按真实输入
                                          通道 lazy 建层）
    - args == [ch, r]    → 保持不变    （位置序与签名一致；ch 与真实输入不符
                                          时同样由 lazy 重建覆盖）
    注意：LCA 签名首参是 channels —— 不做该预处理时 YAML `LCA, [8]` 的 8 会被
    当作 channels、reduction 落回默认 16，导致 r8/r16/r32 消融变体结构相同
    （2026-07-18 修复）。
"""
from __future__ import annotations

import copy

import ultralytics.nn.tasks as tasks_module
from lead_net.models.attention.lca import LCA


def register_lca_to_ultralytics() -> None:
    """注册 LCA + monkey-patch parse_model 对齐 YAML args 与构造签名。"""
    setattr(tasks_module, "LCA", LCA)

    if hasattr(tasks_module, "_lca_patched"):
        return
    tasks_module._lca_patched = True

    original_parse = tasks_module.parse_model

    def patched_parse_model(d, ch, verbose=True):
        """patched：LCA 行单参数 [r] → [None, r]，使 r 落到 reduction 位置。"""
        d = copy.deepcopy(d)
        for section in ("backbone", "head"):
            if section not in d:
                continue
            for row in d[section]:
                if row[2] == "LCA" and len(row[3]) == 1:
                    # [reduction] → [channels=None, reduction]
                    row[3] = [None, row[3][0]]
        return original_parse(d, ch, verbose)

    tasks_module.parse_model = patched_parse_model


# 导入时自动注册
register_lca_to_ultralytics()
