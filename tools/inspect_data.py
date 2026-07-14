"""数据验证脚本：加载几 batch，打印尺寸/类别分布，并保存带框可视化到 outputs/。

依据：
    - docs/DATASET.md §数据全面性检查清单。
    - docs/EXPERIMENTS.md：实验产出可由 outputs/ 落盘。

用法：
    python tools/inspect_data.py --config configs/baseline_ssd.yaml --split val --max-batches 2 --save-dir outputs/data_inspect
    python tools/inspect_data.py --config configs/baseline_ssd.yaml --split train --max-batches 1

需要 COCO 数据已下载解压到 cfg.paths.dataset_root。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lead_net.utils import load_config, resolve_paths_in, get_nested, ensure_dir
from lead_net.data import build_dataloader, build_coco_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net 数据验证")
    p.add_argument("--config", required=True, type=str, help="配置文件路径")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--batch-size", type=int, default=None, help="覆盖 cfg 中的 batch_size（默认小一些=4）")
    p.add_argument("--num-workers", type=int, default=0, help="建议 Windows 用 0")
    p.add_argument("--max-batches", type=int, default=2, help="最多遍历 batch 数")
    p.add_argument("--save-dir", type=str, default="outputs/data_inspect", help="可视化保存目录")
    p.add_argument("--no-save", action="store_true", help="不写盘，仅打印统计")
    return p.parse_args()


def _denormalize(img_t: torch.Tensor, mean, std):
    import numpy as np
    arr = img_t.detach().cpu().float().numpy().transpose(1, 2, 0)  # HWC
    arr = arr * np.array(std) + np.array(mean)
    arr = arr.clip(0.0, 1.0)
    return arr


def save_sample(batch_idx: int, sample_idx: int, image: torch.Tensor, boxes_tv, labels, class_map, save_dir: Path, mean, std):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    arr = _denormalize(image, mean, std)
    ax.imshow(arr)
    if boxes_tv is not None and boxes_tv.numel() > 0:
        from torchvision.tv_tensors import BoundingBoxes
        bxywh = boxes_tv.data if isinstance(boxes_tv, BoundingBoxes) else boxes_tv
        # 可能已变 XYXY 或仍 XYWH（视 transform）；统一按 XYWH 渲染
        for b_idx in range(bxywh.shape[0]):
            x, y, w, h = bxywh[b_idx].tolist()
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=1.5))
            lbl = int(labels[b_idx])
            name = class_map.get(lbl, str(lbl))
            ax.text(x, max(y - 2, 0), name, color="lime", fontsize=8)
    ax.set_title(f"batch{batch_idx}_sample{sample_idx}")
    ax.set_axis_off()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_dir / f"batch{batch_idx}_sample{sample_idx}.png", dpi=100)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = resolve_paths_in(cfg)

    bs = args.batch_size or 4
    loader = build_dataloader(
        cfg,
        split=args.split,
        batch_size=bs,
        num_workers=args.num_workers,
        shuffle=False,
    )
    save_dir = Path(args.save_dir)
    if not args.no_save:
        save_dir = ensure_dir(save_dir)

    class_map: dict = cfg.get("class_map", {})
    mean = cfg["data"]["mean"]
    std = cfg["data"]["std"]
    # CSS inverse for SAVE denorm

    total_imgs = 0
    label_counter: Counter = Counter()
    empty_samples = 0

    print(f"[info] split={args.split} | batch_size={bs} | dataset_root={cfg['paths']['dataset_root']}")
    try:
        for b_idx, batch in enumerate(loader):
            if b_idx >= args.max_batches:
                break
            imgs = batch["image"]
            boxes = batch["boxes"]
            labels = batch["labels"]
            print(f"[batch {b_idx}] image shape={tuple(imgs.shape)} dtype={imgs.dtype} "
                  f"B={imgs.shape[0]}")
            for s in range(imgs.shape[0]):
                total_imgs += 1
                nb = int(boxes[s].shape[0])
                lbls = labels[s].tolist()
                label_counter.update(lbls)
                if nb == 0:
                    empty_samples += 1
                print(f"  - sample {s}: boxes={nb} labels={lbls}")
                if not args.no_save and b_idx < args.max_batches:
                    save_sample(b_idx, s, imgs[s], boxes[s], labels[s], class_map, save_dir, mean, std)
        print(f"[summary] imgs={total_imgs} empty_samples={empty_samples}")
        print(f"[summary] label distribution (internal_id:count) = {dict(label_counter)}")
        if save_dir and not args.no_save:
            print(f"[summary] 可视化已存到 {save_dir}")
    except FileNotFoundError as e:
        print(f"[error] 数据文件未找到：{e}", file=sys.stderr)
        print("[hint] 请先运行 python tools/download_coco.py --root data/coco --split all")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())