"""build_hard_subsets.py — Stage 1Cb 困难场景子集构建工具。

职责（单一）：
    扫描 val 集，按 3 类困难场景筛选图片索引，输出子集索引文件。
    用于 Stage 1Cc LCA 增益验证（在困难子集上对比 FOMO±LCA 的 mAP/recall）。

3 类困难场景：
    1. 小目标: bbox 宽高 < 32px（COCO 小目标定义，input_size=320 时归一化 w*h < (32/320)^2）
       实际用 max(w,h) < 32/320 = 0.1
    2. 遮挡: 无 COCO occlusion 标注，用代理指标——单图目标密集（>5 个目标且互相高度重叠）
       或人工筛选太贵，改用"多目标重叠"代理：图中任意两 bbox IoU > 0.3
    3. 复杂红色背景: 红色通道主导（R > G+B 的像素占比 > 40%）

不负责：
    - 实际训练（Stage 1Cc）
    - LCA 模块（阶段 2）

用法：
    python tools/build_hard_subsets.py
    产出：
      outputs/stage1c/subsets/small_targets.txt
      outputs/stage1c/subsets/occlusion.txt
      outputs/stage1c/subsets/red_background.txt
      outputs/stage1c/subsets/summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


VAL_IMG_DIR = Path("data/lead_subset/images/val")
VAL_LABEL_DIR = Path("data/lead_subset/labels/val")
INPUT_SIZE = 320
OUT_DIR = Path("outputs/stage1c/subsets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 小目标阈值（归一化），COCO 定义 <32×32px，input_size=320 → 32/320=0.1
SMALL_MAX_WH = 32 / INPUT_SIZE  # 0.1

# 遮挡代理：单图目标数 >= 3 且存在两 bbox IoU > 0.3
OCCLUSION_MIN_OBJS = 3
OCCLUSION_IOU_THRESH = 0.3

# 复杂红色背景：R > (G+B) 的像素占比 > 40%
RED_DOMINANT_RATIO = 0.40


def parse_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """解析 YOLO txt 标签：class cx cy w h（归一化）。"""
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(x) for x in parts[1:5])
        boxes.append((cls, cx, cy, w, h))
    return boxes


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """两 bbox（cx,cy,w,h 归一化）的 IoU。"""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter
    return inter / max(union, 1e-9)


def is_small_target(boxes: list[tuple]) -> bool:
    """含至少一个 small target。"""
    for _, _, _, w, h in boxes:
        if max(w, h) < SMALL_MAX_WH:
            return True
    return False


def is_occlusion_proxy(boxes: list[tuple]) -> bool:
    """多目标 + 存在重叠（遮挡代理）。"""
    if len(boxes) < OCCLUSION_MIN_OBJS:
        return False
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou(boxes[i][1:], boxes[j][1:]) > OCCLUSION_IOU_THRESH:
                return True
    return False


def is_red_background(img_path: Path) -> bool:
    """红色通道主导像素占比 > 40%。"""
    try:
        img = Image.open(img_path).convert("RGB").resize((64, 64))
        arr = np.asarray(img, dtype=np.int32)  # [64,64,3]
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        red_dominant = (r > (g + b)).sum()
        ratio = red_dominant / (64 * 64)
        return ratio > RED_DOMINANT_RATIO
    except Exception:
        return False


def main() -> None:
    print("=" * 70)
    print("Stage 1Cb: Building hard-scene subsets")
    print("=" * 70)

    img_paths = sorted(VAL_IMG_DIR.glob("*.jpg")) + sorted(VAL_IMG_DIR.glob("*.png"))
    print(f"val images: {len(img_paths)}")

    small_targets: list[str] = []
    occlusion: list[str] = []
    red_bg: list[str] = []

    for i, img_path in enumerate(img_paths):
        label_path = VAL_LABEL_DIR / (img_path.stem + ".txt")
        boxes = parse_label(label_path)

        if is_small_target(boxes):
            small_targets.append(img_path.name)

        if is_occlusion_proxy(boxes):
            occlusion.append(img_path.name)

        # 红色背景需读图，较慢，每 100 张报告进度
        if is_red_background(img_path):
            red_bg.append(img_path.name)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(img_paths)}] small={len(small_targets)} "
                  f"occ={len(occlusion)} red={len(red_bg)}", flush=True)

    # 写子集索引文件
    (OUT_DIR / "small_targets.txt").write_text("\n".join(small_targets), encoding="utf-8")
    (OUT_DIR / "occlusion.txt").write_text("\n".join(occlusion), encoding="utf-8")
    (OUT_DIR / "red_background.txt").write_text("\n".join(red_bg), encoding="utf-8")

    summary = {
        "val_total": len(img_paths),
        "small_targets": len(small_targets),
        "small_targets_pct": round(100.0 * len(small_targets) / max(len(img_paths), 1), 2),
        "occlusion": len(occlusion),
        "occlusion_pct": round(100.0 * len(occlusion) / max(len(img_paths), 1), 2),
        "red_background": len(red_bg),
        "red_background_pct": round(100.0 * len(red_bg) / max(len(img_paths), 1), 2),
        "thresholds": {
            "small_max_wh": SMALL_MAX_WH,
            "occlusion_min_objs": OCCLUSION_MIN_OBJS,
            "occlusion_iou": OCCLUSION_IOU_THRESH,
            "red_dominant_ratio": RED_DOMINANT_RATIO,
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[done] subsets saved to {OUT_DIR}")
    print(f"  small_targets: {summary['small_targets']} ({summary['small_targets_pct']}%)")
    print(f"  occlusion:     {summary['occlusion']} ({summary['occlusion_pct']}%)")
    print(f"  red_background:{summary['red_background']} ({summary['red_background_pct']}%)")


if __name__ == "__main__":
    main()
