"""attention 模块包。

导出两个 LCA 实现：
    - LCA (新, ultralytics 适配): lead_net.models.attention.lca.LCA
    - build_lca (旧, SSD 路径兼容): 从 attention_legacy 导入
"""
from lead_net.models.attention.lca import LCA, build_lca_from_cfg
from lead_net.models.attention.attention_legacy import build_lca, LCA as LCALegacy

__all__ = ["LCA", "build_lca_from_cfg", "build_lca", "LCALegacy"]
