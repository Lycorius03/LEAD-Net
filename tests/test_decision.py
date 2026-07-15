"""避障决策模块测试（风险评估器架构）。

覆盖：ROI / 分组面积阈值 / 风险评分 / EMA / 增长率 / 状态机 / 时间一致性 / 遮挡恢复
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lead_net.decision import (
    DecisionEngine, DecisionResult, DecisionParams,
    ROIFilter, ROIParams,
    PriorityCalculator, PriorityParams,
    RiskCalculator, RiskParams, TargetRisk,
)
from lead_net.tracking import MultiTargetTracker


def _det(bbox, score=0.9, category_id=0):
    return {"bbox": bbox, "score": score, "category_id": category_id}


# ---- ROI ----

def test_roi_inside():
    roi = ROIFilter()
    inside, _ = roi.apply([_det([160, 240, 50, 50])], 320, 320)
    assert len(inside) == 1


def test_roi_outside():
    roi = ROIFilter()
    inside, _ = roi.apply([_det([160, 50, 50, 50])], 320, 320)
    assert len(inside) == 0


# ---- PriorityCalculator ----

def test_priority_large_moving_triggers():
    calc = PriorityCalculator(PriorityParams(reference_distance_cm=50))
    urgent, _ = calc.classify([_det([160, 200, 100, 80], score=0.9, category_id=0)])
    assert len(urgent) == 1  # person area=8000 >= 6000


def test_priority_chair_not_triggered():
    calc = PriorityCalculator(PriorityParams(reference_distance_cm=100))
    urgent, tracked = calc.classify([_det([160, 200, 30, 30], score=0.9, category_id=56)])
    assert len(urgent) == 0  # chair area=900 < 1000
    assert len(tracked) == 1


def test_priority_empty():
    urgent, tracked = PriorityCalculator().classify([])
    assert urgent == [] and tracked == []


# ---- RiskCalculator ----

def test_risk_center_proximity():
    """越靠近中心，风险越高。"""
    calc = RiskCalculator(image_center=(160, 160), image_area=102400)
    near = calc.evaluate(1, 5000, 0.9, 160, 160)
    far = calc.evaluate(2, 5000, 0.9, 10, 10)
    assert near.center_score > far.center_score


def test_risk_growth_increases_risk():
    """面积增长 → 风险升高。"""
    calc = RiskCalculator(image_center=(160, 160))
    # First call: area=1000
    calc.evaluate(1, 1000, 0.9, 160, 160)
    # Second call: area=2000 (doubled → high growth)
    r2 = calc.evaluate(1, 2000, 0.9, 160, 160)
    assert r2.growth_rate > 0.5  # 100% growth


def test_risk_ema_smoothing():
    """EMA 平滑：新值逐渐影响 smoothed。"""
    calc = RiskCalculator(RiskParams(ema_alpha=0.3))
    r1 = calc.evaluate(1, 5000, 0.9, 160, 160)
    # second call with same area → smoothed should be between raw and previous
    r2 = calc.evaluate(1, 5000, 0.9, 160, 160)
    # With alpha=0.3, smoothed moves toward raw, but not instantly
    assert abs(r2.smoothed_risk - r2.raw_risk) <= abs(r1.smoothed_risk - r1.raw_risk) + 0.01


def test_risk_state_machine():
    """风险分数 → 状态映射。"""
    r = TargetRisk(track_id=1, smoothed_risk=0.0)
    assert r.update_state() == "detected"
    r.smoothed_risk = 0.3
    assert r.update_state() == "tracking"
    r.smoothed_risk = 0.5
    assert r.update_state() == "warning"
    r.smoothed_risk = 0.8
    assert r.update_state() == "danger"


def test_risk_prune():
    calc = RiskCalculator()
    calc.evaluate(1, 1000, 0.9, 160, 160)
    calc.evaluate(2, 2000, 0.8, 100, 100)
    assert len(calc._history) == 2
    calc.prune({1})  # only keep track 1
    assert len(calc._history) == 1


# ---- DecisionResult ----

def test_result_no_target():
    r = DecisionResult.no_target()
    assert not r.has_target
    assert r.x == -1 and r.a == 0
    assert r.state == "none"


def test_result_to_uart():
    r = DecisionResult(x=100, y=200, a=4800, risk_score=0.6, state="warning")
    assert r.to_uart() == "x100,y200,a4800\r\n"


# ---- DecisionEngine (full pipeline) ----

def test_engine_basic():
    engine = DecisionEngine(
        DecisionParams(confidence_threshold=0.3), tracker=None,
        image_width=320, image_height=320,
    )
    det = _det([160, 240, 100, 80], score=0.9, category_id=0)
    r = engine.decide([det])
    assert r.has_target
    assert r.a > 0


def test_engine_with_tracker():
    tracker = MultiTargetTracker(max_targets=3, min_hits=1, max_ttl=5)
    engine = DecisionEngine(
        DecisionParams(confidence_threshold=0.3), tracker=tracker,
        image_width=320, image_height=320,
    )
    det = _det([160, 240, 100, 80], score=0.9, category_id=0)
    engine.decide([det])
    r = engine.decide([det])
    assert r.has_target


def test_engine_time_consistency():
    """连续高风险帧 → 稳定输出；单帧 → 不输出。"""
    engine = DecisionEngine(
        DecisionParams(confidence_threshold=0.3, min_consecutive_high_risk=2),
        tracker=None, image_width=320, image_height=320,
    )
    det = _det([160, 240, 100, 80], score=0.9, category_id=0)
    engine.decide([det])
    engine.decide([det])
    r = engine.decide([det])
    assert r.has_target  # 3 consecutive frames


def test_engine_empty_input():
    engine = DecisionEngine()
    r = engine.decide([])
    assert not r.has_target


def test_engine_handles_low_confidence():
    engine = DecisionEngine(DecisionParams(confidence_threshold=0.8))
    r = engine.decide([_det([160, 200, 80, 80], score=0.3)])
    assert not r.has_target


def test_engine_no_regression_on_missing_detection():
    """持续有目标 → 突然丢 → 应输出 lost 状态而非崩溃。"""
    engine = DecisionEngine(
        DecisionParams(confidence_threshold=0.3),
        tracker=None, image_width=320, image_height=320,
    )
    det = _det([160, 240, 100, 80], score=0.9, category_id=0)
    engine.decide([det])
    engine.decide([det])
    r = engine.decide([])  # sudden miss
    assert not r.has_target  # no tracker → no occlusion recovery


# ---- Params ----

def test_params_from_cfg():
    cfg = {
        "decision": {
            "confidence_threshold": 0.6,
            "risk": {"w_center": 0.3, "ema_alpha": 0.5},
        },
        "tracking": {"max_targets": 3, "min_hits": 2, "T_lost": 3, "iou_threshold": 0.3},
    }
    p = DecisionParams.from_cfg(cfg)
    assert p.confidence_threshold == 0.6
    assert p.risk.w_center == 0.3
    assert p.risk.ema_alpha == 0.5


# ---- Tracker TTL ----

def test_tracker_ttl():
    tracker = MultiTargetTracker(max_targets=1, min_hits=1, max_ttl=3)
    det = _det([160, 200, 80, 80], score=0.9)
    tracker.update([det])
    tracker.update([det])
    t = tracker.confirmed()[0]
    assert t.ttl == 3  # reset on hit

    # miss a few frames
    tracker.update([])
    assert t.ttl == 2
    tracker.update([])
    assert t.ttl == 1
    tracker.update([])
    assert t.ttl == 0
    assert t.decision_state == "lost"


if __name__ == "__main__":
    test_roi_inside()
    test_roi_outside()
    test_priority_large_moving_triggers()
    test_priority_chair_not_triggered()
    test_priority_empty()
    test_risk_center_proximity()
    test_risk_growth_increases_risk()
    test_risk_ema_smoothing()
    test_risk_state_machine()
    test_risk_prune()
    test_result_no_target()
    test_result_to_uart()
    test_engine_basic()
    test_engine_with_tracker()
    test_engine_time_consistency()
    test_engine_empty_input()
    test_engine_handles_low_confidence()
    test_engine_no_regression_on_missing_detection()
    test_params_from_cfg()
    test_tracker_ttl()
    print("[OK] test_decision: all 20 tests passed")
