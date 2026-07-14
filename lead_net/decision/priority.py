"""紧急度（Priority）计算器 —— 三层决策流程第三层。

基于 bbox 面积代理（距离反比）+ 置信度 + 类别危险等级加权。

理论依据：
    - DisNet (Haseeb et al.): bbox 尺寸直接作为距离回归特征
    - DECADE (Shahzad et al., 2024): 不生成像素级深度图，纯 bbox 几何推理
    - J-MOD² (Mancini et al., arXiv:1709.08480): bbox 深度与检测联合学习

优先级公式:
    priority = w_area * area_norm + w_conf * confidence + w_class * class_danger[cls]
    area_norm = bbox_area / max_area_in_frame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# 默认类别危险等级（0.0-1.0，越高越危险，越需优先避让）
_DEFAULT_CLASS_DANGER: dict[int, float] = {
    # person 类（COCO internal id）
    0: 1.0,    # person — 最高优先级
    # 车辆类
    2: 0.9,    # car
    3: 0.85,   # motorcycle
    5: 0.85,   # bus
    7: 0.85,   # truck
    1: 0.7,    # bicycle
    # 动物类
    14: 0.7,   # bird
    15: 0.7,   # cat
    16: 0.7,   # dog
    17: 0.7,   # horse
    18: 0.7,   # sheep
    19: 0.7,   # cow
    20: 0.7,   # elephant
    21: 0.7,   # bear
    22: 0.7,   # zebra
    23: 0.7,   # giraffe
    # 大型家具（可能挡路）
    56: 0.5,   # chair
    57: 0.5,   # couch
    58: 0.5,   # potted plant
    59: 0.5,   # bed
    60: 0.5,   # dining table
    61: 0.5,   # toilet
    # 箱包类
    24: 0.3,   # backpack
    26: 0.3,   # handbag
    28: 0.3,   # suitcase
    # 运动器材
    30: 0.3,   # skis
    31: 0.3,   # snowboard
    36: 0.3,   # skateboard
    37: 0.3,   # surfboard
    38: 0.3,   # tennis racket
    34: 0.3,   # baseball bat
    35: 0.3,   # baseball glove
    # 未知/CV检测 → 中等优先级（保守策略）
    -1: 0.5,
}

# 默认：未在表中列出的类别
_DEFAULT_DANGER = 0.2


@dataclass
class PriorityParams:
    """优先级计算参数。"""

    area_weight: float = 0.6
    confidence_weight: float = 0.3
    class_weight: float = 0.1
    min_priority: float = 0.2      # 低于此值视为不紧急
    class_danger: dict[int, float] = field(default_factory=lambda: dict(_DEFAULT_CLASS_DANGER))
    default_danger: float = 0.2

    @classmethod
    def from_cfg(cls, cfg: dict) -> "PriorityParams":
        d = cfg.get("decision", {})
        return cls(
            area_weight=d.get("area_weight", 0.6),
            confidence_weight=d.get("confidence_weight", 0.3),
            class_weight=d.get("class_weight", 0.1),
            min_priority=d.get("min_priority", 0.2),
        )


class PriorityCalculator:
    """紧急度计算器。

    用法::

        calc = PriorityCalculator(params, image_area=320*320)
        scored = calc.compute(detections)
        best = calc.select_top(scored)
    """

    def __init__(self, params: PriorityParams | None = None,
                 image_area: float = 102400.0):
        self.params = params or PriorityParams()
        self.image_area = image_area  # 图像总面积 (px²)，用于面积归一化

    def compute(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """为每个检测计算 priority 分数并附加到 dict。

        面积归一化：bbox_area / image_area（物理一致，避免单检测帧永远归一化为1）。

        Returns:
            detections 列表，每个 dict 新增 "priority" 字段，按 priority 降序排列。
        """
        if not detections:
            return []

        scored = []
        for det in detections:
            bbox = det["bbox"]
            area = bbox[2] * bbox[3]
            # 归一化到全图面积：典型障碍物占画面 5%-30%，对应 area_norm 0.05-0.30
            area_norm = area / max(self.image_area, 1.0)

            cls_id = det.get("category_id", -1)
            danger = self.params.class_danger.get(
                cls_id, self.params.default_danger,
            )

            priority = (
                self.params.area_weight * area_norm * 10.0  # ×10 缩放至与 conf/danger 同量级
                + self.params.confidence_weight * det.get("score", 0.0)
                + self.params.class_weight * danger
            )

            det["priority"] = priority
            scored.append(det)

        scored.sort(key=lambda d: d["priority"], reverse=True)
        return scored

    def select_top(
        self, scored: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """取最高优先级目标，低于阈值返回 None。"""
        if not scored:
            return None
        top = scored[0]
        if top.get("priority", 0.0) < self.params.min_priority:
            return None
        return top
