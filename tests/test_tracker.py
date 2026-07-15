"""多目标追踪器单元测试。

覆盖：Track 生命周期、贪心匹配、优先级选择、边界情况、输出格式。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lead_net.tracking import MultiTargetTracker, Track


def _det(bbox, score=0.9, category_id=0):
    return {"bbox": bbox, "score": score, "category_id": category_id}


def test_empty_update():
    tracker = MultiTargetTracker()
    tracks = tracker.update([])
    assert tracks == []
    assert tracker.select_priority() is None
    assert tracker.all_active() == []


def test_tentative_to_confirmed():
    tracker = MultiTargetTracker(min_hits=2)
    d = _det([100, 120, 60, 80], score=0.9, category_id=2)

    # 第1帧：创建 tentative (hits=1)
    tracker.update([d])
    assert len(tracker.all_active()) == 1
    assert tracker.all_active()[0].state == "tentative"
    assert len(tracker.confirmed()) == 0

    # 第2帧：连续命中 → confirmed
    tracker.update([d])
    assert len(tracker.confirmed()) == 1
    assert tracker.confirmed()[0].state == "confirmed"
    assert tracker.confirmed()[0].hits == 2


def test_tentative_quick_removal():
    """tentative 目标一帧未匹配即被移除。"""
    tracker = MultiTargetTracker(min_hits=2)
    tracker.update([_det([100, 120, 60, 80])])
    assert len(tracker.all_active()) == 1
    # 下一帧无匹配
    tracker.update([])
    assert len(tracker.all_active()) == 0


def test_confirmed_lost_timeout():
    """confirmed 目标丢失超过 T_lost+TTL 帧后删除。"""
    tracker = MultiTargetTracker(min_hits=2, T_lost=3, max_ttl=1)
    d = _det([100, 120, 60, 80])
    tracker.update([d])
    tracker.update([d])
    assert len(tracker.confirmed()) == 1

    # TTL=1→0 (1 frame), then T_lost=3 more → total 4 frames
    for _ in range(5):
        tracker.update([])
    assert len(tracker.all_active()) == 0


def test_greedy_matching_two_targets():
    """两个检测应对应两个独立 track。"""
    tracker = MultiTargetTracker(min_hits=1)  # 1 hit → immediate confirm
    d1 = _det([80, 80, 40, 40], score=0.9, category_id=0)
    d2 = _det([200, 100, 50, 60], score=0.8, category_id=2)
    tracker.update([d1, d2])
    confirmed = tracker.confirmed()
    assert len(confirmed) == 2
    ids = {t.id for t in confirmed}
    assert len(ids) == 2


def test_priority_center_distance():
    """离画面中心近的目标优先级更高。"""
    tracker = MultiTargetTracker(min_hits=1, image_center=(160, 160))
    d_far = _det([80, 80, 40, 40])       # dist ~113
    d_near = _det([200, 140, 50, 60])     # dist ~45
    tracker.update([d_far, d_near])
    top = tracker.select_priority()
    assert top is not None
    # d_near 离中心更近
    s = top.kf.state()
    assert abs(s[0] - 200) < 10, f"expected near center cx, got {s[0]:.0f}"


def test_priority_tiebreaker_area():
    """中心距离相同时，面积更大的优先。"""
    tracker = MultiTargetTracker(min_hits=1, image_center=(160, 160))
    d_small = _det([160, 160, 20, 20])
    d_large = _det([160, 160, 60, 60])
    tracker.update([d_small, d_large])
    top = tracker.select_priority()
    s = top.kf.state()
    # 面积更大 → w*h=3600 vs 400
    assert s[2] * s[3] > 1000


def test_iou_matching_threshold():
    """IoU 低于阈值的检测不匹配已有 track，应创建新 track。"""
    tracker = MultiTargetTracker(min_hits=1, iou_threshold=0.5)
    d1 = _det([80, 80, 50, 50])
    d2 = _det([200, 200, 50, 50])  # IoU with d1 = 0
    tracker.update([d1])
    tracker.update([d2])
    # d1 track 不匹配 d2，应创建新 track
    active = tracker.all_active()
    assert len(active) == 2


def test_format_stm32():
    tracker = MultiTargetTracker(min_hits=1)
    tracker.update([_det([100, 120, 60, 80])])
    top = tracker.select_priority()
    msg = MultiTargetTracker.format_stm32(top)
    assert msg == "x100,y120,a48\r\n"

    msg_none = MultiTargetTracker.format_stm32(None)
    assert msg_none == "x-1,y-1,a0\r\n"


def test_format_experiment():
    tracker = MultiTargetTracker(min_hits=1)
    tracker.update([_det([200, 100, 50, 60])])
    top = tracker.select_priority()
    msg = MultiTargetTracker.format_experiment(top, 12345)
    assert msg == "t:12345, x:200, y:100, a:30\n"


def test_max_targets_prune():
    """超过 max_targets 时，低优先级 target 被移除。"""
    tracker = MultiTargetTracker(max_targets=2, min_hits=1, image_center=(160, 160))
    # 3 个目标，中心的那个优先级最高
    tracker.update([
        _det([80, 80, 40, 40]),     # far, lowest priority
        _det([140, 140, 30, 30]),   # medium
        _det([160, 160, 50, 50]),   # center, highest
    ])
    confirmed = tracker.confirmed()
    assert len(confirmed) <= 2


def test_track_to_dict():
    tracker = MultiTargetTracker(min_hits=1)
    tracker.update([_det([120, 100, 60, 80], score=0.95, category_id=3)])
    top = tracker.select_priority()
    d = top.to_dict()
    assert d["id"] == 1
    assert d["state"] == "confirmed"
    assert d["cls"] == 3
    assert abs(d["conf"] - 0.95) < 0.01
    for key in ("cx", "cy", "w", "h", "vx", "vy", "vw", "vh"):
        assert key in d


if __name__ == "__main__":
    test_empty_update()
    test_tentative_to_confirmed()
    test_tentative_quick_removal()
    test_confirmed_lost_timeout()
    test_greedy_matching_two_targets()
    test_priority_center_distance()
    test_priority_tiebreaker_area()
    test_iou_matching_threshold()
    test_format_stm32()
    test_format_experiment()
    test_max_targets_prune()
    test_track_to_dict()
    print("[OK] test_tracker: all tests passed")
