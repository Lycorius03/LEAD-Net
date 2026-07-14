"""避障决策模块测试。

覆盖：ROI 过滤 / priority 计算 / DL-CV 融合 / 三层全流程 / 边界情况。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lead_net.decision import (
    DecisionEngine, DecisionResult, DecisionParams,
    ROIFilter, ROIParams,
    PriorityCalculator, PriorityParams,
    FusionStrategy, FusionParams,
)


def _det(bbox, score=0.9, category_id=0):
    return {"bbox": bbox, "score": score, "category_id": category_id}


# ---- ROI Filter ----

def test_roi_inside():
    roi = ROIFilter(ROIParams(h_start=0.2, h_end=0.8, v_start=0.4, v_end=1.0))
    inside, outside = roi.apply(
        [_det([160, 200, 50, 50])],  # center in ROI
        image_width=320, image_height=320,
    )
    assert len(inside) == 1 and len(outside) == 0


def test_roi_outside_horizontal():
    roi = ROIFilter(ROIParams(h_start=0.5, h_end=0.8, v_start=0.0, v_end=1.0))
    inside, outside = roi.apply(
        [_det([30, 160, 50, 50])],  # far left
        image_width=320, image_height=320,
    )
    assert len(inside) == 0 and len(outside) == 1


def test_roi_outside_vertical():
    roi = ROIFilter(ROIParams(h_start=0.0, h_end=1.0, v_start=0.6, v_end=1.0))
    inside, outside = roi.apply(
        [_det([160, 50, 50, 50])],  # top of image
        image_width=320, image_height=320,
    )
    assert len(inside) == 0 and len(outside) == 1


# ---- Priority Calculator ----

def test_priority_larger_area_higher():
    calc = PriorityCalculator(
        PriorityParams(area_weight=1.0, confidence_weight=0.0, class_weight=0.0),
    )
    scored = calc.compute([
        _det([100, 100, 30, 30], score=0.9),   # area=900
        _det([200, 200, 80, 80], score=0.9),   # area=6400 → should be #1
    ])
    assert scored[0]["bbox"][2] == 80  # larger area wins


def test_priority_confidence_boost():
    calc = PriorityCalculator(
        PriorityParams(area_weight=0.0, confidence_weight=1.0, class_weight=0.0),
    )
    scored = calc.compute([
        _det([100, 100, 50, 50], score=0.5),
        _det([100, 100, 50, 50], score=0.95),
    ])
    assert scored[0]["score"] == 0.95


def test_priority_person_over_chair():
    calc = PriorityCalculator(
        PriorityParams(
            area_weight=0.0, confidence_weight=0.0, class_weight=1.0,
        ),
    )
    scored = calc.compute([
        _det([100, 100, 50, 50], score=0.9, category_id=56),   # chair (danger=0.5)
        _det([100, 100, 50, 50], score=0.9, category_id=0),    # person (danger=1.0)
    ])
    assert scored[0]["category_id"] == 0  # person > chair


def test_priority_below_threshold():
    calc = PriorityCalculator(
        PriorityParams(
            area_weight=0.0, confidence_weight=1.0, class_weight=0.0,
            min_priority=0.8,
        ),
    )
    scored = calc.compute([_det([100, 100, 50, 50], score=0.3)])
    top = calc.select_top(scored)
    assert top is None  # priority=0.3 < 0.8


def test_priority_empty():
    calc = PriorityCalculator()
    assert calc.compute([]) == []
    assert calc.select_top([]) is None


# ---- Fusion Strategy ----

def test_fusion_dl_cv_overlap():
    fusion = FusionStrategy(FusionParams(iou_threshold=0.3))
    dl = [_det([100, 100, 50, 50], score=0.9)]
    cv = [{"bbox": [105, 102, 48, 52], "score": 0.6}]  # high overlap
    merged = fusion.merge(dl, cv)
    assert len(merged) == 1
    assert merged[0]["source"] == "FUSION"
    assert merged[0]["score"] == 0.9  # max of 0.9 and 0.6


def test_fusion_dl_only():
    fusion = FusionStrategy()
    dl = [_det([100, 100, 50, 50], score=0.9)]
    cv = [{"bbox": [250, 250, 20, 20], "score": 0.5}]  # no overlap
    merged = fusion.merge(dl, cv)
    assert len(merged) == 2  # DL + CV added separately
    sources = {m["source"] for m in merged}
    assert sources == {"DL", "CV"}


def test_fusion_cv_only_new():
    fusion = FusionStrategy()
    dl: list = []
    cv = [{"bbox": [150, 200, 40, 40], "score": 0.7}]
    merged = fusion.merge(dl, cv)
    assert len(merged) == 1
    assert merged[0]["source"] == "CV"
    assert merged[0]["category_id"] == -1  # UNKNOWN


def test_fusion_empty():
    fusion = FusionStrategy()
    assert fusion.merge([], []) == []


# ---- DecisionEngine (三层全流程) ----

def test_engine_full_flow_detects_obstacle():
    engine = DecisionEngine(
        DecisionParams(confidence_threshold=0.3),
        image_width=320, image_height=320,
    )
    result = engine.decide([
        _det([160, 220, 80, 80], score=0.9, category_id=0),   # person, center, large
    ])
    assert result is not None
    assert result.class_id == 0
    assert result.source == "DL"
    assert result.priority > 0
    assert result.layer_log["confidence"] is True
    assert result.layer_log["roi"] is True
    assert result.layer_log["priority"] is True


def test_engine_filters_low_confidence():
    engine = DecisionEngine(
        DecisionParams(confidence_threshold=0.8),
    )
    result = engine.decide([
        _det([160, 200, 80, 80], score=0.2),  # below threshold
    ])
    assert result is None
    assert engine.stats["layer1_filtered"] == 1


def test_engine_filters_outside_roi():
    engine = DecisionEngine(
        DecisionParams(
            confidence_threshold=0.3,
            roi=ROIParams(h_start=0.3, h_end=0.7, v_start=0.5, v_end=1.0),
        ),
        image_width=320, image_height=320,
    )
    result = engine.decide([
        _det([30, 300, 50, 50], score=0.9),  # far left, not in ROI
    ])
    assert result is None
    assert engine.stats["layer2_filtered"] == 1


def test_engine_filters_low_priority():
    engine = DecisionEngine(
        DecisionParams(
            confidence_threshold=0.3,
            priority=PriorityParams(
                area_weight=1.0, confidence_weight=0.0, class_weight=0.0,
                min_priority=0.9,
            ),
        ),
    )
    result = engine.decide([
        _det([160, 200, 10, 10], score=0.9),  # tiny area → low priority
    ])
    assert result is None


def test_engine_with_cv_fusion():
    engine = DecisionEngine(
        DecisionParams(confidence_threshold=0.3),
        image_width=320, image_height=320,
    )
    dl = [_det([160, 200, 60, 60], score=0.7)]
    cv = [{"bbox": [160, 200, 60, 60], "score": 0.5}]
    result = engine.decide(dl, cv)
    assert result is not None
    assert result.source == "FUSION"
    assert engine.stats["cv_contributions"] == 1


def test_engine_empty_input():
    engine = DecisionEngine()
    result = engine.decide([])
    assert result is None


def test_engine_stats_accumulate():
    engine = DecisionEngine()
    engine.decide([_det([160, 200, 80, 80], score=0.9)])
    engine.decide([_det([30, 30, 20, 20], score=0.9)])  # should be filtered by ROI
    assert engine.stats["total_frames"] == 2
    assert engine.stats["decisions_made"] == 1
    assert engine.stats["layer2_filtered"] >= 1


def test_engine_reset_stats():
    engine = DecisionEngine()
    engine.decide([_det([160, 200, 80, 80], score=0.9)])
    engine.reset_stats()
    assert engine.stats["total_frames"] == 0


def test_decision_result_to_dict():
    r = DecisionResult(x=100, y=200, w=50, h=60, priority=0.85,
                       class_id=0, source="FUSION",
                       layer_log={"confidence": True, "roi": True, "priority": True})
    d = r.to_dict()
    assert d["bbox"] == [100, 200, 50, 60]
    assert d["priority"] == 0.85
    assert d["source"] == "FUSION"
    assert d["layer_log"]["priority"] is True


def test_params_from_cfg():
    cfg = {
        "decision": {
            "confidence_threshold": 0.5,
            "roi_horizontal_range": [0.2, 0.8],
            "roi_vertical_range": [0.3, 1.0],
            "area_weight": 0.5,
            "min_priority": 0.3,
        }
    }
    params = DecisionParams.from_cfg(cfg)
    assert params.confidence_threshold == 0.5
    assert params.roi.h_start == 0.2
    assert params.roi.v_start == 0.3
    assert params.priority.area_weight == 0.5
    assert params.priority.min_priority == 0.3


if __name__ == "__main__":
    test_roi_inside()
    test_roi_outside_horizontal()
    test_roi_outside_vertical()
    test_priority_larger_area_higher()
    test_priority_confidence_boost()
    test_priority_person_over_chair()
    test_priority_below_threshold()
    test_priority_empty()
    test_fusion_dl_cv_overlap()
    test_fusion_dl_only()
    test_fusion_cv_only_new()
    test_fusion_empty()
    test_engine_full_flow_detects_obstacle()
    test_engine_filters_low_confidence()
    test_engine_filters_outside_roi()
    test_engine_filters_low_priority()
    test_engine_with_cv_fusion()
    test_engine_empty_input()
    test_engine_stats_accumulate()
    test_engine_reset_stats()
    test_decision_result_to_dict()
    test_params_from_cfg()
    print("[OK] test_decision: all 22 tests passed")
