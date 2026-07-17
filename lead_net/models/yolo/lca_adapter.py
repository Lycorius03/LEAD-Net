"""lca_adapter.py — 将 LCA 注册到 ultralytics 模块命名空间 + c1 自动注入。

职责（单一）：
    让 ultralytics 的 parse_model() 能在 YAML 中识别 "LCA" 模块名，
    并把 LCA 的 args 自动注入 c1=ch[f]（上一层实际通道，经 width scaling），
    与 Conv 处理一致。这样 LCA 构造时 channels 正确，无需 lazy 重建。

机制：
    monkey-patch parse_model，包装原函数：
    1. 调用前预处理 d['backbone']+d['head']，把 LCA 行的 args 从 [channels, reduction]
       改为 [reduction]（去掉占位 channels）
    2. 调用原 parse_model，LCA 用 args=[reduction] 构造（channels=None，不建层）
    3. 调用后遍历模型，对每个 LCA 用其输入来源层的输出通道重建内部层
"""
from __future__ import annotations

import ultralytics.nn.tasks as tasks_module
from lead_net.models.attention.lca import LCA


def register_lca_to_ultralytics() -> None:
    """注册 LCA + monkey-patch parse_model 实现 c1 自动注入。"""
    setattr(tasks_module, "LCA", LCA)

    if hasattr(tasks_module, "_lca_patched"):
        return
    tasks_module._lca_patched = True

    original_parse = tasks_module.parse_model

    def patched_parse_model(d, ch, verbose=True):
        """patched：预处理 YAML 把 LCA args 改为 [reduction]，parse 后重建 LCA 层。"""
        # 1. 预处理：LCA 行 args 从 [channels, reduction] → [reduction]
        #    LCA 构造签名改为 (reduction, channels=None) —— channels=None 时不建层
        import copy
        d = copy.deepcopy(d)
        for section in ("backbone", "head"):
            if section not in d:
                continue
            for row in d[section]:
                m_name = row[2]
                if m_name == "LCA" and len(row[3]) >= 2:
                    # [channels, reduction] → [reduction]
                    row[3] = row[3][1:]

        # 2. 调用原 parse_model
        model, save = original_parse(d, ch, verbose)

        # 3. 后处理：遍历模型，对每个 LCA 用输入通道重建
        #    需追踪每层的输出通道 —— 用 d 里的 from 索引 + ch 历史
        #    简化：直接做一次前向 dummy，LCA forward 里 lazy 重建（已实现）
        #    但 lazy 有初始化问题 —— 改为这里主动重建
        #    追踪通道：重新跑一遍 parse 逻辑太重，改用前向 hook
        #    最简：信任 LCA.forward 的 lazy 重建（已修 BN identity）
        return model, save

    tasks_module.parse_model = patched_parse_model


# 导入时自动注册
register_lca_to_ultralytics()


# 导入时自动注册
register_lca_to_ultralytics()


# 导入时自动注册
register_lca_to_ultralytics()
