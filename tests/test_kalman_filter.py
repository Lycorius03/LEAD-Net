"""Kalman 滤波器单元测试。

覆盖：初始化 / predict / update / 收敛性 / 边界值。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from lead_net.tracking import KalmanFilter


def test_init():
    kf = KalmanFilter(dt=1.0)
    kf.init(160, 160, 50, 50)
    assert kf.is_initialized
    s = kf.state()
    assert s.shape == (8,)
    assert s[0] == 160 and s[1] == 160 and s[2] == 50 and s[3] == 50
    # 初始速度为零
    assert all(s[4:] == 0)


def test_predict_no_update():
    kf = KalmanFilter(dt=1.0)
    kf.init(100, 100, 40, 40)
    s = kf.predict()
    # 速度为零时，位置不变
    assert abs(s[0] - 100) < 1e-6
    assert abs(s[1] - 100) < 1e-6


def test_update_pulls_state():
    kf = KalmanFilter(dt=1.0)
    kf.init(100, 100, 50, 50)
    kf.predict()
    # 观测偏右
    kf.update(np.array([120.0, 100.0, 50.0, 50.0], dtype=np.float64))
    s = kf.state()
    # 状态应向 120 方向修正
    assert s[0] > 100 and s[0] < 120


def test_velocity_estimation():
    """匀速移动目标：Kalman 应学习到速度。"""
    kf = KalmanFilter(dt=1.0)
    kf.init(100, 100, 50, 50)
    for i in range(20):
        kf.predict()
        # 每帧向右移动 3px
        z = np.array([100 + (i + 1) * 3, 100, 50, 50], dtype=np.float64)
        kf.update(z)
    vx = kf.state()[4]
    # 速度应收敛到 ~3
    assert 2.0 < vx < 4.0, f"vx={vx:.2f}, expected ~3.0"


def test_smoothing_effect():
    """Kalman 应对噪声观测有平滑效果。"""
    kf = KalmanFilter(dt=1.0)
    kf.init(160, 160, 50, 50)
    states = []
    rng = np.random.RandomState(42)
    for _ in range(20):
        kf.predict()
        noise = rng.randn(4) * 5  # std=5 噪声
        z = np.array([160 + noise[0], 160 + noise[1],
                       50 + noise[2], 50 + noise[3]], dtype=np.float64)
        kf.update(z)
        states.append(kf.state()[:4].copy())
    # 平滑后的位置方差应小于观测噪声方差
    positions = np.array(states)
    var_cx = positions[:, 0].var()
    # 观测噪声 ~25 (5²)，平滑后应显著小于
    assert var_cx < 20, f"var_cx={var_cx:.1f}, expected < 20"


def test_dt_effect():
    """dt 影响预测步长。"""
    kf = KalmanFilter(dt=2.0)
    kf.init(100, 100, 50, 50)
    # 手动设速度
    kf._x[4] = 5.0
    kf.predict()
    s = kf.state()
    # dt=2: cx += 5*2 = 10
    assert abs(s[0] - 110) < 0.01


def test_zero_size_box():
    """零尺寸检测框不应崩溃。"""
    kf = KalmanFilter(dt=1.0)
    kf.init(160, 160, 0, 0)
    kf.predict()
    kf.update(np.array([160.0, 160.0, 0.1, 0.1], dtype=np.float64))
    s = kf.state()
    assert s[2] > 0 and s[3] > 0


if __name__ == "__main__":
    test_init()
    test_predict_no_update()
    test_update_pulls_state()
    test_velocity_estimation()
    test_smoothing_effect()
    test_dt_effect()
    test_zero_size_box()
    print("[OK] test_kalman_filter: all tests passed")
