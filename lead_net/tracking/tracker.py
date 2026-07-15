"""多目标 Kalman 追踪器。

Track 数据结构 + MultiTargetTracker（贪心 IoU 匹配 + 生命周期管理 + 优先级选择）。

参考：
    - SORT (arXiv:1602.00763)：tracking-by-detection 框架
    - 本项目简化：N=3 贪心匹配（非 Hungarian）、单目标优先级输出

设计决策（写入 MODULES.md §4）：
    - max_targets=3：对应小车前方左/中/右三方向，平衡计算开销与覆盖
    - 贪心匹配（非 Hungarian）：N≤3 时全排列 ≤6 种组合，贪心计算量极小
    - T_lost=3：比 SORT 默认 1 宽松，适配低帧率/遮挡恢复
    - min_hits=2：2 帧连续命中确认轨迹，减少单帧误检
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .kalman_filter import KalmanFilter


@dataclass
class Track:
    """单个追踪目标的状态容器。

    Attributes:
        id: 全局自增 ID（从 1 开始）。
        kf: 独立的 Kalman 滤波器实例。
        state: "tentative" | "confirmed" | "deleted".
        age: 自创建以来的总帧数。
        time_since_update: 自上次成功匹配以来的帧数。
        hits: 连续命中次数（tentative → confirmed 的判断依据）。
        cls: 最后匹配的类别标签（0-based internal id）。
        conf: 最后匹配的检测置信度。
        history: 最近 N 帧的 (cx, cy, w, h) 记录（用于实验分析）。
    """

    id: int
    kf: KalmanFilter
    state: str = "tentative"
    age: int = 0
    time_since_update: int = 0
    hits: int = 0
    ttl: int = 5              # 目标生命周期：丢失多少帧后降级为 lost
    decision_state: str = "detected"  # detected|tracking|warning|danger|lost
    cls: int = -1
    conf: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def area(self) -> float:
        s = self.kf.state()
        return float(s[2] * s[3])

    @property
    def velocity(self) -> tuple[float, float, float, float]:
        """返回 (vx, vy, vw, vh) 供增长率计算。"""
        s = self.kf.state()
        return (float(s[4]), float(s[5]), float(s[6]), float(s[7]))

    def predict_position(self, steps: int = 1) -> tuple[float, float]:
        """预测 steps 帧后的 (cx, cy) 位置（遮挡恢复）。"""
        s = self.kf.state()
        return (float(s[0] + s[4] * steps), float(s[1] + s[5] * steps))

    def to_dict(self) -> dict[str, Any]:
        s = self.kf.state()
        return {
            "id": self.id,
            "cx": float(s[0]), "cy": float(s[1]),
            "w": float(s[2]), "h": float(s[3]),
            "vx": float(s[4]), "vy": float(s[5]),
            "vw": float(s[6]), "vh": float(s[7]),
            "state": self.state,
            "decision_state": self.decision_state,
            "cls": self.cls,
            "conf": self.conf,
            "age": self.age,
            "time_since_update": self.time_since_update,
            "ttl": self.ttl,
        }


class MultiTargetTracker:
    """多目标 Kalman 追踪器。

    每帧调用 update(detections) 进行预测-匹配-更新-生命周期管理。
    通过 select_priority() 获取优先级最高的目标用于串口输出。

    Args:
        max_targets: 最大同时追踪目标数（默认 3）。
        min_hits: tentative → confirmed 所需连续命中帧数（默认 2）。
        T_lost: confirmed 丢失多少帧后删除（默认 3）。
        iou_threshold: 匹配最低 IoU 阈值（默认 0.3）。
        kalman_dt: Kalman 帧间时间间隔（默认 1.0）。
        image_center: 图像中心 (cx, cy)，用于优先级排序（默认 160,160）。
    """

    def __init__(
        self,
        max_targets: int = 3,
        min_hits: int = 2,
        T_lost: int = 3,
        max_ttl: int = 5,
        iou_threshold: float = 0.3,
        kalman_dt: float = 1.0,
        image_center: tuple[float, float] = (160.0, 160.0),
    ):
        self.max_targets = max_targets
        self.min_hits = min_hits
        self.max_ttl = max_ttl
        self.T_lost = T_lost
        self.iou_threshold = iou_threshold
        self.kalman_dt = kalman_dt
        self.image_center = image_center

        self._next_id = 1
        self._tracks: list[Track] = []

    # ---- public API ----

    @property
    def tracks(self) -> list[Track]:
        """返回当前所有非 deleted 的 tracks。"""
        return [t for t in self._tracks if t.state != "deleted"]

    def update(self, detections: list[dict]) -> list[Track]:
        """处理一帧检测结果，返回当前 confirmed tracks。

        Args:
            detections: list of dict, 每个 dict 含
                "bbox": [x, y, w, h]（xywh 绝对像素坐标）、
                "score": float、
                "category_id": int（0-based internal id）。

        Returns:
            当前 confirmed 状态的 tracks 列表（已按优先级排序）。
        """
        # 1. 所有活跃 track 预测
        for t in self._tracks:
            if t.state != "deleted":
                t.kf.predict()
                t.age += 1

        # 2. 贪心 IoU 匹配
        active = [t for t in self._tracks if t.state != "deleted"]
        matches, unmatched_dets, unmatched_tracks = self._greedy_match(
            detections, active,
        )

        # 3. 匹配成功 → 观测更新
        for det_idx, track_idx, _iou in matches:
            det = detections[det_idx]
            bbox = det["bbox"]
            z = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float64)
            t = active[track_idx]
            t.kf.update(z)
            t.time_since_update = 0
            t.hits += 1
            t.ttl = self.max_ttl   # 重置 TTL
            t.decision_state = "tracking"
            t.cls = det.get("category_id", -1)
            t.conf = det.get("score", 0.0)
            s = t.kf.state()
            t.history.append((float(s[0]), float(s[1]), float(s[2]), float(s[3])))
            if t.state == "tentative" and t.hits >= self.min_hits:
                t.state = "confirmed"

        # 4. 未匹配检测 → 创建新 tentative track
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            bbox = det["bbox"]
            kf = KalmanFilter(dt=self.kalman_dt)
            kf.init(bbox[0], bbox[1], bbox[2], bbox[3])
            track = Track(
                id=self._next_id, kf=kf, cls=det.get("category_id", -1),
                conf=det.get("score", 0.0), hits=1,
            )
            track.ttl = self.max_ttl
            if track.hits >= self.min_hits:
                track.state = "confirmed"
            self._next_id += 1
            s = kf.state()
            track.history.append((float(s[0]), float(s[1]), float(s[2]), float(s[3])))
            self._tracks.append(track)

        # 5. 未匹配 track → 标记丢失，TTL 递减
        for track_idx in unmatched_tracks:
            t = active[track_idx]
            t.time_since_update += 1
            t.ttl = max(0, t.ttl - 1)
            if t.ttl == 0:
                t.decision_state = "lost"

        # 6. 生命周期：deleted 清理
        self._cleanup()

        # 7. 限制活跃 track 数
        self._prune()

        return self.confirmed()

    def get_lost_predictions(self) -> list[Track]:
        """返回已丢失但仍在 Kalman 预测中的 tracks（遮挡恢复）。"""
        return [t for t in self._tracks if t.decision_state == "lost"
                and t.state != "deleted"]

    def confirmed(self) -> list[Track]:
        """返回所有 confirmed 状态的 tracks，按优先级排序。"""
        c = [t for t in self._tracks if t.state == "confirmed"]
        return sorted(c, key=lambda t: self._priority_key(t))

    def select_priority(self) -> Track | None:
        """返回优先级最高的 confirmed track（用于 STM32 串口输出）。"""
        confirmed = self.confirmed()
        return confirmed[0] if confirmed else None

    def all_active(self) -> list[Track]:
        """返回所有活跃 tracks（tentative + confirmed），用于实验数据记录。"""
        return [t for t in self._tracks if t.state in ("tentative", "confirmed")]

    # ---- 输出格式 ----
    # UART 协议输出统一走 DecisionResult.to_uart()（decision_engine.py），
    # 此处不再重复实现，避免 area 语义不一致（如 /100 vs 原始面积）。
    # USB 实验日志使用 USBLogger（decision_engine.py）。

    # ---- internal ----

    def _priority_key(self, track: Track) -> tuple[float, float]:
        """优先级排序键：(中心距离↑, 面积↓)。距离越小优先级越高。"""
        s = track.kf.state()
        cx, cy, w, h = s[0], s[1], s[2], s[3]
        dist = np.sqrt((cx - self.image_center[0]) ** 2 + (cy - self.image_center[1]) ** 2)
        area = w * h
        return (dist, -area)

    def _greedy_match(
        self, detections: list[dict], tracks: list[Track],
    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        """贪心 IoU 匹配。

        Returns:
            matches: [(det_idx, track_idx, iou), ...]
            unmatched_dets: [det_idx, ...]
            unmatched_tracks: [track_idx, ...]
        """
        if not detections or not tracks:
            return [], list(range(len(detections))), list(range(len(tracks)))

        N_d, N_t = len(detections), len(tracks)
        iou_mat = np.zeros((N_d, N_t), dtype=np.float64)
        for i in range(N_d):
            for j in range(N_t):
                iou_mat[i, j] = self._box_iou(
                    detections[i]["bbox"], self._track_bbox(tracks[j]),
                )

        # 按 IoU 降序取配对
        pairs = []
        for i in range(N_d):
            for j in range(N_t):
                if iou_mat[i, j] > self.iou_threshold:
                    pairs.append((iou_mat[i, j], i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])

        used_d = set()
        used_t = set()
        matches = []
        for iou, di, ti in pairs:
            if di not in used_d and ti not in used_t:
                matches.append((di, ti, iou))
                used_d.add(di)
                used_t.add(ti)

        unmatched_dets = [i for i in range(N_d) if i not in used_d]
        unmatched_tracks = [j for j in range(N_t) if j not in used_t]
        return matches, unmatched_dets, unmatched_tracks

    @staticmethod
    def _track_bbox(track: Track) -> list[float]:
        s = track.kf.state()
        return [float(s[0]), float(s[1]), float(s[2]), float(s[3])]

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

    def _cleanup(self) -> None:
        """删除超时或 marked deleted 的 tracks。

        TTL 机制：lost 状态时保留 Kalman 预测，允许遮挡恢复。
        仅当 TTL=0 且超过 T_lost 时才真正删除。
        """
        new = []
        for t in self._tracks:
            if t.state == "deleted":
                continue
            if t.state == "tentative" and t.time_since_update > 0:
                continue
            # confirmed track: TTL 耗尽 + 超时 → 删除
            if t.state == "confirmed":
                if t.ttl == 0 and t.time_since_update > self.T_lost:
                    continue
            new.append(t)
        self._tracks = new

    def _prune(self) -> None:
        """当 confirmed 数超过 max_targets 时，移除优先级最低的。"""
        confirmed = [t for t in self._tracks if t.state == "confirmed"]
        if len(confirmed) > self.max_targets:
            confirmed.sort(key=lambda t: self._priority_key(t))
            # 移除优先级最低的（排序最靠后的）
            to_remove = set(t.id for t in confirmed[self.max_targets:])
            self._tracks = [t for t in self._tracks if t.id not in to_remove]
