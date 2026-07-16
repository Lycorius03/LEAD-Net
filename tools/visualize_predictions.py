#!/usr/bin/env python
"""BBox 可视化诊断工具 —— 加载模型，在验证图片上画出GT框和预测框。

用途：
    - 诊断 mAP=0 问题时，可视化检查预测框是否贴合物体
    - 验证 bbox 解码坐标是否正确
    - 检查 NMS 是否工作正常

用法：
    python tools/visualize_predictions.py \
        --config configs/train_baseline.yaml \
        --weights outputs/checkpoints/last.pth \
        --num-images 10 \
        --score-threshold 0.01

输出：
    outputs/pred_viz/ 目录下的 PNG 图片
        - 绿色框：Ground Truth
        - 红色框：Prediction（附带类别名+置信度）

设计原则：
    - 独立于训练/评估管线，零副作用
    - 可指定置信度阈值，方便调试低置信度问题
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from lead_net.data import build_dataloader
from lead_net.engine.evaluator import _build_coco_gt_from_dataset
from lead_net.models import build_lead_net
from lead_net.utils import load_config, resolve_paths_in, get_nested, ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BBox 可视化诊断")
    p.add_argument("--config", required=True, help="配置文件路径")
    p.add_argument("--weights", required=True, help="checkpoint 路径")
    p.add_argument("--num-images", type=int, default=10, help="可视化图片数")
    p.add_argument("--score-threshold", type=float, default=0.01,
                   help="置信度阈值（调试用建议0.01）")
    p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--output-dir", type=str, default="outputs/pred_viz")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def denormalize(img_t: torch.Tensor, mean, std) -> np.ndarray:
    """反归一化 ImageNet 标准化到 [0,1] HWC numpy."""
    arr = img_t.detach().cpu().float().numpy().transpose(1, 2, 0)
    arr = arr * np.array(std) + np.array(mean)
    return arr.clip(0.0, 1.0)


def draw_boxes(ax, boxes_xywh, labels, scores, class_map, color, linewidth=2.0):
    """在 matplotlib ax 上画框。

    Args:
        boxes_xywh: [N, 4] 像素坐标 (x, y, w, h)
        labels: [N] 类别 ID (0-based)
        scores: [N] 置信度（None 表示GT框）
        class_map: {id: name}
    """
    import matplotlib.patches as patches

    for i in range(len(boxes_xywh)):
        x, y, w, h = boxes_xywh[i].tolist() if isinstance(boxes_xywh, torch.Tensor) else boxes_xywh[i]
        if w <= 0 or h <= 0:
            continue
        rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor=color,
                                 linewidth=linewidth)
        ax.add_patch(rect)
        lbl_id = int(labels[i]) if isinstance(labels, torch.Tensor) else labels[i]
        name = class_map.get(lbl_id, class_map.get(str(lbl_id), f"cls_{lbl_id}"))
        if scores is not None:
            sc = float(scores[i]) if isinstance(scores, torch.Tensor) else scores[i]
            label_text = f"{name} {sc:.2f}"
        else:
            label_text = f"GT:{name}"
        ax.text(x, max(y - 2, 0), label_text, color=color, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = resolve_paths_in(cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[viz] device: {device}")

    # 数据加载（不 shuffle，取前 N 张）
    val_loader = build_dataloader(cfg, split="val", num_workers=0, shuffle=False)
    class_map = cfg.get("class_map", {})
    input_size = cfg.get("data", {}).get("input_size", 320)
    mean = cfg["data"]["mean"]
    std = cfg["data"]["std"]

    # 模型加载
    model = build_lead_net(cfg).to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[viz] model loaded from {args.weights}")

    # 输出目录
    out_dir = ensure_dir(Path(args.output_dir))
    print(f"[viz] output dir: {out_dir}")

    # 获取 GT 信息用于对比
    dataset = val_loader.dataset
    has_coco = hasattr(dataset, "coco")

    img_count = 0
    nms_cfg = cfg.get("eval", {}).get("nms", {})
    max_det = nms_cfg.get("max_detections", 100)
    pre_nms_topk = nms_cfg.get("pre_nms_topk", 1000)

    for batch in val_loader:
        if img_count >= args.num_images:
            break

        images = batch["image"].to(device)
        gt_boxes_list = batch["boxes"]
        gt_labels_list = batch["labels"]

        with torch.no_grad():
            cls_pred, loc_pred = model(images)
            detections = model.head.decode(
                loc_pred, cls_pred,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_iou,
                max_detections=max_det,
                pre_nms_topk=pre_nms_topk,
            )

        for i in range(images.size(0)):
            if img_count >= args.num_images:
                break

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

            # 反归一化图像
            img_np = denormalize(images[i], mean, std)

            # --- 左图：GT 框 ---
            ax1.imshow(img_np)
            ax1.set_title(f"Ground Truth (img {img_count})")
            ax1.set_axis_off()

            gt_boxes = gt_boxes_list[i]
            gt_labels = gt_labels_list[i]
            if len(gt_boxes) > 0:
                # GT boxes 可能已由 transform 转为 resize 后的坐标
                draw_boxes(ax1, gt_boxes, gt_labels, None, class_map, "lime", linewidth=2.0)
            ax1.text(5, 15, f"{len(gt_boxes)} GT boxes", color="lime", fontsize=9,
                     bbox=dict(facecolor="black", alpha=0.7))

            # --- 右图：预测框 ---
            ax2.imshow(img_np)
            ax2.set_title(f"Predictions (score>{args.score_threshold}, NMS={args.nms_iou})")
            ax2.set_axis_off()

            dets = detections[i]
            if dets:
                pred_boxes = torch.tensor([d["bbox"] for d in dets])
                pred_labels = [d["category_id"] for d in dets]
                pred_scores = [d["score"] for d in dets]
                draw_boxes(ax2, pred_boxes, pred_labels, pred_scores, class_map, "red", linewidth=1.5)
                # 同时画 GT 作为参考
                if len(gt_boxes) > 0:
                    draw_boxes(ax2, gt_boxes, gt_labels, None, class_map, "lime", linewidth=1.0)
                ax2.text(5, 15, f"{len(dets)} preds (score>{args.score_threshold})",
                         color="red", fontsize=9, bbox=dict(facecolor="black", alpha=0.7))
            else:
                ax2.text(160, 160, "NO PREDICTIONS", color="red", fontsize=14,
                         ha="center", bbox=dict(facecolor="black", alpha=0.7))
                print(f"[viz] img {img_count}: 0 predictions at score_thr={args.score_threshold}")

            # 打印诊断信息
            if dets:
                scores = [d["score"] for d in dets]
                print(f"[viz] img {img_count}: {len(dets)} preds, "
                      f"score range [{min(scores):.4f}, {max(scores):.4f}], "
                      f"GT boxes: {len(gt_boxes)}")
            else:
                print(f"[viz] img {img_count}: NO preds, GT boxes: {len(gt_boxes)}")

            fig.tight_layout()
            fig.savefig(out_dir / f"pred_viz_{img_count:03d}.png", dpi=120)
            plt.close(fig)

            img_count += 1

    print(f"\n[viz] done. {img_count} images saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
