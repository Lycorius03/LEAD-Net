"""传统 CV 兜底模块 —— 地面分割 + blob 检测 + 前景掩膜。

对应 PLAN §M9，服务 RQ6b（传统 CV 对 DL 漏检的补充效果）。

提供：
    - CvFallback：完整 CV 前景检测管道（分割→前景→blob）
    - GroundSegmenter：基于颜色聚类的地面/可通行区域分割
    - BlobDetector：连通分量标记 + 边界框提取

参考：
    - R18 (StixelNExT++, 2025): 10ms 自由空间检测
    - R19 (UAV自监督系统, 2025): DL+几何混合
    - R20 (IJCV 2024): 地面分割抗反光
"""

from .ground_segmenter import GroundSegmenter, GroundSegmenterParams
from .blob_detector import BlobDetector, BlobParams
from .foreground_mask import CvFallback, CvFallbackParams

__all__ = [
    "CvFallback",
    "CvFallbackParams",
    "GroundSegmenter",
    "GroundSegmenterParams",
    "BlobDetector",
    "BlobParams",
]
