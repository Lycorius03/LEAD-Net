"""DL/CV 融合策略 —— 区域重叠验证。

将 DL 检测框与传统 CV 前景掩膜区域进行 IoU 匹配，实现：
    - DL+CV 双重确认 → 提高 priority（高可信度）
    - DL 检出但 CV 未验证 → 降低 priority（可能误检）
    - CV 检出但 DL 漏检 → 补充为 UNKNOWN 类检测框

融合后的检测框统一送入三层决策流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FusionParams:
    """融合策略参数。"""

    iou_threshold: float = 0.3       # 最低 IoU 判定为"匹配"
    dl_only_penalty: float = 0.7     # DL-only 的 priority 折扣因子
    cv_only_confidence: float = 0.4  # CV-only 新增框的默认置信度
    cv_only_class_id: int = -1       # CV-only 框的类别 ID（UNKNOWN）

    @classmethod
    def from_cfg(cls, cfg: dict) -> "FusionParams":
        d = cfg.get("decision", {})
        return cls(
            iou_threshold=d.get("fusion_iou_threshold", 0.3),
            dl_only_penalty=d.get("fusion_dl_penalty", 0.7),
            cv_only_confidence=d.get("fusion_cv_confidence", 0.4),
        )


class FusionStrategy:
    """DL/CV 检测结果融合。

    用法::

        fusion = FusionStrategy(params)
        merged = fusion.merge(dl_detections, cv_regions, img_w, img_h)
    """

    def __init__(self, params: FusionParams | None = None):
        self.params = params or FusionParams()

    def merge(
        self,
        dl_detections: list[dict[str, Any]],
        cv_regions: list[dict[str, Any]],
        image_width: int = 320,
        image_height: int = 320,
    ) -> list[dict[str, Any]]:
        """融合 DL 检测与 CV 前景区域。

        Args:
            dl_detections: LEAD-Net 输出 [{"bbox": [cx,cy,w,h], "score":, "category_id":}, ...]
            cv_regions: 传统 CV 前景掩膜 [{"bbox": [cx,cy,w,h], "score":}, ...]
            image_width, image_height: 图像尺寸（CV bbox 可能需归一化）

        Returns:
            融合后的检测列表，每个含 "source" 字段标记来源（DL/CV/FUSION）
        """
        # 标记 DL 来源
        for det in dl_detections:
            det.setdefault("source", "DL")

        # 标记 CV 来源
        for region in cv_regions:
            region.setdefault("source", "CV")
            region.setdefault("category_id", self.params.cv_only_class_id)
            region.setdefault("score", self.params.cv_only_confidence)

        if not cv_regions:
            return dl_detections
        if not dl_detections:
            return cv_regions

        merged = []
        matched_cv = set()

        for dl_det in dl_detections:
            best_iou = 0.0
            best_cv_idx = -1
            for j, cv_reg in enumerate(cv_regions):
                iou = self._box_iou(dl_det["bbox"], cv_reg["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_cv_idx = j

            if best_iou >= self.params.iou_threshold and best_cv_idx not in matched_cv:
                # DL + CV 双重确认 → 提高置信度
                merged_det = dict(dl_det)
                merged_det["source"] = "FUSION"
                merged_det["score"] = max(dl_det.get("score", 0), cv_regions[best_cv_idx].get("score", 0))
                merged_det["fusion_iou"] = best_iou
                merged.append(merged_det)
                matched_cv.add(best_cv_idx)
            else:
                # DL-only → 降低 priority（可能误检）
                penalized = dict(dl_det)
                penalized["source"] = "DL"
                penalized["score"] = dl_det.get("score", 0) * self.params.dl_only_penalty
                merged.append(penalized)

        # CV-only 区域（DL 漏检） → 补充为 UNKNOWN
        for j, cv_reg in enumerate(cv_regions):
            if j not in matched_cv:
                merged.append({
                    "bbox": cv_reg["bbox"],
                    "score": cv_reg.get("score", self.params.cv_only_confidence),
                    "category_id": self.params.cv_only_class_id,
                    "source": "CV",
                })

        return merged

    @staticmethod
    def _box_iou(a: list[float], b: list[float]) -> float:
        """cxcywh 格式两个框的 IoU。"""
        ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
        ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
        bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
        bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = a[2] * a[3]
        area_b = b[2] * b[3]
        union = area_a + area_b - inter
        return inter / union if union > 1e-8 else 0.0
