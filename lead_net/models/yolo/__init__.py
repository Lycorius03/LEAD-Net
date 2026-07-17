"""yolo 模块包：LCA 适配 + YOLO11 构建 + 数据适配。"""
# 导入即注册 LCA 到 ultralytics
from lead_net.models.yolo import lca_adapter  # noqa: F401

__all__ = ["lca_adapter"]
