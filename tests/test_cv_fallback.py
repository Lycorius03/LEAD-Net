"""传统 CV 兜底模块测试。

覆盖：地面分割 / blob 检测 / 前景掩膜管道 / 边界情况。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lead_net.cv_fallback import (
    GroundSegmenter, GroundSegmenterParams,
    BlobDetector, BlobParams,
    CvFallback, CvFallbackParams,
)


def _make_uniform_image(h=64, w=64, color=(120, 140, 100)):
    """创建纯色图像。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def _add_obstacle(img, x, y, w, h, color=(40, 30, 20)):
    """在图像上添加障碍物矩形。"""
    img[y:y+h, x:x+w] = color


# ---- GroundSegmenter ----

def test_ground_uniform():
    """均匀颜色图像：全部应为地面。"""
    img = _make_uniform_image()
    seg = GroundSegmenter(GroundSegmenterParams(sample_ratio=0.3, color_threshold=40))
    mask = seg.segment(img)
    assert mask.sum() > 0.9 * mask.size, f"Ground ratio: {mask.sum()/mask.size:.3f}"


def test_ground_with_obstacle():
    """底部绿色地面 + 上部深色障碍物：障碍物区域应为非地面。"""
    img = _make_uniform_image(color=(120, 140, 100))  # 绿调地面
    _add_obstacle(img, 16, 8, 20, 20, color=(40, 30, 20))  # 深色障碍物
    seg = GroundSegmenter(GroundSegmenterParams(sample_ratio=0.3, color_threshold=40))
    mask = seg.segment(img)
    # 障碍物区域（左上20x20）应该被标记为非地面
    obstacle_region = mask[8:28, 16:36]
    obstacle_ratio = obstacle_region.sum() / obstacle_region.size
    assert obstacle_ratio < 0.5, f"Obstacle misclassified as ground: {obstacle_ratio:.2f}"


def test_ground_dominant_color():
    """验证主导色计算正确。"""
    img = _make_uniform_image(color=(100, 150, 200))
    seg = GroundSegmenter()
    seg.segment(img)
    dominant = seg.dominant_color
    assert dominant is not None
    assert abs(dominant[0] - 100) < 2
    assert abs(dominant[1] - 150) < 2
    assert abs(dominant[2] - 200) < 2


def test_ground_empty_image():
    """零尺寸图像不崩溃。"""
    seg = GroundSegmenter()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = seg.segment(img)
    assert mask.shape == (10, 10)


# ---- BlobDetector ----

def test_blob_single_rectangle():
    """单个矩形前景 → 1 个 blob。"""
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 15:35] = True  # 20x20 矩形
    detector = BlobDetector(BlobParams(min_area=10))
    blobs = detector.detect(mask)
    assert len(blobs) == 1
    bbox = blobs[0]["bbox"]
    assert 14 <= bbox[0] <= 36  # cx near 25
    assert 9 <= bbox[1] <= 31   # cy near 20
    assert blobs[0]["area"] == 400


def test_blob_multiple():
    """两个分离前景 → 2 个 blob。"""
    mask = np.zeros((64, 64), dtype=bool)
    mask[5:15, 5:15] = True    # 10x10
    mask[40:55, 40:55] = True  # 15x15
    detector = BlobDetector(BlobParams(min_area=20))
    blobs = detector.detect(mask)
    assert len(blobs) == 2


def test_blob_min_area_filter():
    """面积小于阈值的 blob 被过滤。"""
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:13, 10:13] = True  # 3x3=9 pixels
    detector = BlobDetector(BlobParams(min_area=50))
    blobs = detector.detect(mask)
    assert len(blobs) == 0


def test_blob_empty():
    """空掩膜 → 空列表。"""
    mask = np.zeros((64, 64), dtype=bool)
    detector = BlobDetector()
    assert detector.detect(mask) == []


# ---- CvFallback (full pipeline) ----

def test_cv_full_pipeline():
    """完整管道：地面+障碍物图像 → 前景 → blob 区域。"""
    img = _make_uniform_image(h=64, w=64, color=(120, 140, 100))
    _add_obstacle(img, 20, 15, 24, 24, color=(40, 30, 20))
    cv = CvFallback()
    regions = cv.process(img)
    # 应该检测到至少1个障碍物区域
    assert len(regions) >= 1, f"Expected >=1 obstacle, got {len(regions)}"
    for r in regions:
        assert "bbox" in r
        assert "score" in r
        assert r["bbox"][2] > 0 and r["bbox"][3] > 0  # w,h > 0
        assert 0.0 <= r["score"] <= 1.0
        # area = w × h（bbox 面积，与 DL 模块语义对齐）
        expected_area = int(round(r["bbox"][2] * r["bbox"][3]))
        assert r["area"] == expected_area, \
            f"area mismatch: {r['area']} != {expected_area} (w={r['bbox'][2]}, h={r['bbox'][3]})"
        # pixel_count 保留原始 blob 像素数供调试
        assert "pixel_count" in r


def test_cv_uniform_ground():
    """纯地面 → 无前景 → 0 个障碍物。"""
    img = _make_uniform_image(color=(120, 140, 100))
    cv = CvFallback(CvFallbackParams(
        ground=GroundSegmenterParams(sample_ratio=0.3, color_threshold=40)))
    regions = cv.process(img)
    assert len(regions) == 0


def test_cv_masks_accessible():
    """验证 ground/foreground mask 可访问。"""
    img = _make_uniform_image()
    cv = CvFallback()
    cv.process(img)
    assert cv.ground_mask is not None
    assert cv.foreground_mask is not None
    assert cv.dominant_color is not None


def test_cv_with_target_size():
    """target_size 参数：将 bbox 坐标从物理分辨率缩放到目标空间。"""
    img = _make_uniform_image(h=64, w=64, color=(120, 140, 100))
    _add_obstacle(img, 20, 15, 24, 24, color=(40, 30, 20))
    cv = CvFallback()
    # target_size: 64→320 缩放比例=5x
    regions = cv.process(img, target_size=(320, 320))
    assert len(regions) >= 1
    for r in regions:
        cx, cy, w, h = r["bbox"]
        # 原始障碍物约在 (32, 26)，缩放后应约在 (160, 130)
        assert 100 <= cx <= 220, f"scaled cx out of range: {cx}"
        assert 80 <= cy <= 180, f"scaled cy out of range: {cy}"
        assert w > 0 and h > 0
        # area 也应同步缩放
        assert r["area"] == int(round(w * h)), \
            f"scaled area mismatch: {r['area']} != {int(round(w * h))}"


if __name__ == "__main__":
    test_ground_uniform()
    test_ground_with_obstacle()
    test_ground_dominant_color()
    test_ground_empty_image()
    test_blob_single_rectangle()
    test_blob_multiple()
    test_blob_min_area_filter()
    test_blob_empty()
    test_cv_full_pipeline()
    test_cv_uniform_ground()
    test_cv_masks_accessible()
    test_cv_with_target_size()
    print("[OK] test_cv_fallback: all 12 tests passed")
