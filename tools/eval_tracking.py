"""追踪稳定性评估工具 —— 帧间方差 / ID切换 / 恢复时间。

用途（对应 RQ4）：
    - 对连续帧运行检测+追踪，记录追踪质量指标
    - 有/无 Kalman 对比（同一序列跑两次）
    - 不同遮挡程度下的表现

用法：
    python tools/eval_tracking.py --config configs/baseline_ssd.yaml \
        --weights outputs/checkpoints/baseline_no_lca.pth \
        --image-dir data/coco/val2017 --max-frames 100
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from lead_net.models import build_lead_net
from lead_net.data.transforms import build_transforms
from lead_net.tracking import MultiTargetTracker
from lead_net.utils import load_config, get_nested, ExperimentManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net 追踪稳定性评估")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--weights", required=True, type=str)
    p.add_argument("--image-dir", required=True, type=str,
                   help="连续帧图像目录（按文件名排序）")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--no-kalman", action="store_true",
                   help="禁用 Kalman 追踪（作为对照组）")
    p.add_argument("--output", type=str, default="outputs/experiments/tracking_stability.csv")
    return p.parse_args()


def load_model(cfg: dict, weights_path: str, device: torch.device):
    model = build_lead_net(cfg)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_detection(model, image: Image.Image, transforms, device, cfg):
    """对单张图片运行检测，返回检测列表。"""
    sample = {"image": image, "boxes": torch.zeros((0, 4)), "labels": torch.tensor([])}
    sample = transforms(sample)
    img_tensor = sample["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        cls_pred, loc_pred = model(img_tensor)

    eval_cfg = cfg.get("eval", {})
    detections = model.head.decode(
        loc_pred, cls_pred,
        score_threshold=eval_cfg.get("score_threshold", 0.05),
        nms_threshold=eval_cfg.get("nms", {}).get("iou_threshold", 0.45),
        max_detections=eval_cfg.get("nms", {}).get("max_detections", 100),
        pre_nms_topk=eval_cfg.get("nms", {}).get("pre_nms_topk", 1000),
    )
    return detections[0]  # list of dicts


def compute_iou(a: dict, b: dict) -> float:
    """cxcywh 格式两个检测框的 IoU。"""
    ax, ay, aw, ah = a["bbox"]
    bx, by, bw, bh = b["bbox"]
    ax1, ay1 = ax - aw/2, ay - ah/2
    ax2, ay2 = ax + aw/2, ay + ah/2
    bx1, by1 = bx - bw/2, by - bh/2
    bx2, by2 = bx + bw/2, by + bh/2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 1e-8 else 0.0


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = cfg.get("data", {}).get("input_size", 320)
    tag = get_nested(cfg, "experiment.tag", "model")

    # 加载模型
    model = load_model(cfg, args.weights, device)
    transforms = build_transforms(cfg, split="val")
    input_size_t = (input_size, input_size)

    # 加载图像序列
    image_dir = Path(args.image_dir)
    images = sorted(image_dir.glob("*.jpg"))[:args.max_frames]
    if not images:
        print(f"[error] no images in {image_dir}")
        return 1
    print(f"[info] {len(images)} frames from {image_dir}")

    # 追踪器
    tracker = None if args.no_kalman else MultiTargetTracker(
        max_targets=3, min_hits=2, T_lost=3, max_ttl=5,
    )

    # ---- 逐帧运行 ----
    prev_centers: dict[int, tuple[float, float]] = {}  # track_id → (cx, cy)
    center_shifts: list[float] = []       # 帧间中心位移
    id_switches: int = 0
    total_recoveries: int = 0
    prev_track_ids: set[int] = set()
    last_seen: dict[int, int] = {}        # track_id → last frame seen
    recovery_times: list[int] = []        # 恢复帧数

    # 记录每帧最佳检测（无 Kalman 对照组）
    best_det_positions: list[dict] = []

    for frame_idx, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        img = img.resize(input_size_t)

        detections = run_detection(model, img, transforms, device, cfg)

        if tracker is not None:
            tracker.update(detections)
            confirmed = tracker.confirmed()

            # 统计当前帧活跃 track IDs
            current_ids = {t.id for t in confirmed}
            new_ids = current_ids - prev_track_ids
            lost_ids = prev_track_ids - current_ids

            # ID 切换：有新 ID 出现而旧 ID 失踪（可能切换了）
            if new_ids and lost_ids:
                id_switches += min(len(new_ids), len(lost_ids))

            # 跟踪中心位移
            for t in confirmed:
                s = t.kf.state()
                cx, cy = float(s[0]), float(s[1])
                if t.id in prev_centers:
                    dx = cx - prev_centers[t.id][0]
                    dy = cy - prev_centers[t.id][1]
                    center_shifts.append(np.sqrt(dx**2 + dy**2))
                prev_centers[t.id] = (cx, cy)
                last_seen[t.id] = frame_idx

            # 恢复时间统计
            for tid in new_ids:
                if tid in last_seen:
                    gap = frame_idx - last_seen[tid]
                    if gap > 1:
                        recovery_times.append(gap)
                        total_recoveries += 1

            prev_track_ids = current_ids

        else:
            # 无 Kalman：记录每帧最佳检测位置
            if detections:
                best = max(detections, key=lambda d: d["score"])
                best_det_positions.append({
                    "frame": frame_idx,
                    "cx": best["bbox"][0],
                    "cy": best["bbox"][1],
                    "w": best["bbox"][2],
                    "h": best["bbox"][3],
                })

    # ---- 汇总指标 ----
    tag_full = f"{tag}_{'no_kalman' if args.no_kalman else 'with_kalman'}"

    if center_shifts:
        mean_shift = np.mean(center_shifts)
        std_shift = np.std(center_shifts)
        max_shift = np.max(center_shifts)
    else:
        mean_shift = std_shift = max_shift = 0.0

    mean_recovery = np.mean(recovery_times) if recovery_times else 0.0

    print(f"\n=== Tracking Stability: {tag_full} ===")
    print(f"  Frames processed:      {len(images)}")
    print(f"  Mean center shift:     {mean_shift:.2f} px")
    print(f"  Std center shift:      {std_shift:.2f} px")
    print(f"  Max center shift:      {max_shift:.2f} px")
    print(f"  ID switches:           {id_switches}")
    print(f"  Recoveries:            {total_recoveries}")
    print(f"  Mean recovery time:    {mean_recovery:.1f} frames")

    # ---- CSV (to variant tracking/ subdir) ----
    variant = "lca" if "lca" in tag.lower() else "baseline"
    mgr = ExperimentManager.for_test("outputs/experiments", variant)
    csv_path = mgr.tracking_dir / f"{tag_full}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        if not file_exists:
            f.write("tag,frames,mean_shift,std_shift,max_shift,id_switches,recoveries,mean_recovery_frames\n")
        f.write(f"{tag_full},{len(images)},{mean_shift:.2f},{std_shift:.2f},{max_shift:.2f},{id_switches},{total_recoveries},{mean_recovery:.1f}\n")

    print(f"[info] results → {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
