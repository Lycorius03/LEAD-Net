"""紧急度判定器 —— 三层决策流程第三层（重写版）。

基于"危险分组 + 面积阈值"而非加权公式：
    - 大型/移动组（person, car, bicycle...）：低阈值 → 更远就触发
    - 中小型/静止组（chair, bottle, backpack...）：高阈值 → 允许更近
    - 面积阈值来源：实测标定表（30cm/50cm/100cm → 类别 → 面积 px²）

设计原则（来自最终避障决策方案）：
    检测精细化，决策通用化。不依赖"是否认出具体类别"来判断"算不算障碍"，
    只依赖"有没有东西、在不在路径上、多近"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---- 危险分组 ----

# 大型/移动组：阈值更低 → 更远即判定紧急
_GROUP_LARGE_MOVING: set[int] = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
}

# 中小型/静止组：阈值更高 → 允许更近距离
_GROUP_SMALL_STATIC: set[int] = {
    24,  # backpack
    26,  # handbag
    28,  # suitcase
    39,  # bottle
    40,  # wine glass
    41,  # cup
    44,  # spoon
    56,  # chair
    57,  # couch
    58,  # potted plant
    60,  # dining table
}

# 默认：未在以上两组的类别使用中等阈值
_DEFAULT_GROUP = "default"


def _group_for(cls_id: int) -> str:
    if cls_id in _GROUP_LARGE_MOVING:
        return "large_moving"
    if cls_id in _GROUP_SMALL_STATIC:
        return "small_static"
    return _DEFAULT_GROUP


# ---- 默认标定表（占位，待实测后替换） ----
# 格式：{group: {distance_cm: area_px²_threshold}}
# 含义：在给定距离下，该类物体的典型 bbox 面积应超过此阈值
# 例：person 在 50cm 处面积约 14400px²（@320×320），阈值设为 8000
_DEFAULT_CALIBRATION: dict[str, dict[int, float]] = {
    "large_moving": {
        30:  20000.0,   # 30cm 处大型移动物体 ≥ 20000px² → 紧急
        50:   6000.0,
        100:  1500.0,
    },
    "small_static": {
        30:  12000.0,
        50:   4000.0,
        100:  1000.0,
    },
    "default": {
        30:  15000.0,
        50:   5000.0,
        100:  1200.0,
    },
}


@dataclass
class PriorityParams:
    """紧急度判定参数。

    不再使用加权公式。改用双组面积阈值 + 标定表。
    """

    # 标定表：{group_name: {distance_cm: area_px²_threshold}}
    calibration: dict[str, dict[int, float]] = field(
        default_factory=lambda: dict(_DEFAULT_CALIBRATION),
    )

    # 触发的参考距离（cm）。取标定表中 ≤ 此距离的阈值中最宽松的。
    # 例：reference_distance=50 → 使用 50cm 行的阈值
    reference_distance_cm: int = 50

    # 单目测距备用预案开关（默认关闭）
    use_monocular_distance: bool = False

    # 类别平均真实尺寸（仅 use_monocular_distance=True 时使用）
    class_real_size: dict[int, float] = field(default_factory=dict)

    # 相机焦距 px（仅 use_monocular_distance=True 时使用）
    camera_focal_length_px: float = 300.0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "PriorityParams":
        d = cfg.get("decision", {})
        return cls(
            calibration=d.get("calibration", _DEFAULT_CALIBRATION),
            reference_distance_cm=d.get("reference_distance_cm", 50),
            use_monocular_distance=d.get("use_monocular_distance", False),
            class_real_size=d.get("class_real_size", {}),
            camera_focal_length_px=d.get("camera_focal_length_px", 300.0),
        )


class PriorityCalculator:
    """基于面积阈值的紧急度判定器。

    用法::

        calc = PriorityCalculator(params)
        urgent, tracked = calc.classify(detections)
    """

    def __init__(self, params: PriorityParams | None = None):
        self.params = params or PriorityParams()

    def classify(
        self, detections: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """将检测结果分为"紧急"和"跟踪中"两组。

        Args:
            detections: [{"bbox":[cx,cy,w,h], "score":, "category_id":}, ...]

        Returns:
            (urgent, tracked):
                urgent  — 超过面积阈值的紧急目标
                tracked — 未超阈值，保留记录用于追踪，不参与紧急判定
        """
        urgent, tracked = [], []
        ref_dist = self.params.reference_distance_cm

        for det in detections:
            bbox = det["bbox"]
            area = bbox[2] * bbox[3]
            cls_id = det.get("category_id", -1)

            group = _group_for(cls_id)
            calib = self.params.calibration.get(group, self.params.calibration.get("default", {}))

            # 取 ≤ reference_distance 的阈值中最宽松的（最小值）
            threshold = min(
                (v for d, v in calib.items() if d <= ref_dist),
                default=float("inf"),
            )

            # 单目测距修正（若启用，将阈值按估计距离缩放）
            if self.params.use_monocular_distance and cls_id in self.params.class_real_size:
                real_size = self.params.class_real_size[cls_id]
                # 针孔模型：estim_dist = (real_size * focal) / sqrt(area)
                import math
                estim = (real_size * self.params.camera_focal_length_px) / max(math.sqrt(area), 1.0)
                # 若估计距离 > 参考距离，收紧阈值（提高门槛）
                if estim > ref_dist:
                    threshold *= (estim / ref_dist) ** 2

            if area >= threshold:
                det["priority"] = area / max(threshold, 1.0)
                det["danger_group"] = group
                urgent.append(det)
            else:
                det["priority"] = 0.0
                det["danger_group"] = group
                tracked.append(det)

        # urgent 按面积降序（大=近=优先）
        urgent.sort(key=lambda d: d.get("priority", 0), reverse=True)
        return urgent, tracked
