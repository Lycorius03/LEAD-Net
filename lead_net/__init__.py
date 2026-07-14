"""LEAD-Net 源码包。

模块组织：
    models/    Backbone / Detection Head / 组装
    data/      Dataset / Transforms / DataLoader
    engine/    Trainer / Evaluator
    tracking/  Kalman Filter / Multi-target Tracker
    utils/     config / path 等工具

设计原则见 docs/ARCHITECTURE.md。
"""

__version__ = "0.1.0-m4"