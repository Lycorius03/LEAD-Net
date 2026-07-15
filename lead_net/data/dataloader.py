"""DataLoader 工厂 + 检测 collate。

依据：
    - docs/MODULES.md §5 训练与量化 Pipeline：DataLoader 与训练流程相关。
    - docs/ARCHITECTURE.md §配置驱动：num_workers/batch_size 从 cfg["train"] 读。

设计细节（M1 第二步 a）：
    - Dataset.__getitem__ 返回 dict(image, boxes, labels, image_id, ...)，长度可变；
      用自定义 collate 把 image 堆为 [B,3,H,W]，boxes/labels 保持 list[Tensor]（变长）。
    - Windows 下 DataLoader num_workers!=0 可能触发问题；cfg 默认值已在 baseline_ssd.yaml 标注待调。
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

from .coco_dataset import build_coco_dataset


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """检测 collate：图像堆叠，boxes/labels 保留 list。"""
    images = torch.stack([b["image"] for b in batch], dim=0)  # [B,3,H,W]
    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    image_ids = torch.stack([b["image_id"] for b in batch], dim=0)
    return {
        "image": images,
        "boxes": boxes,
        "labels": labels,
        "image_id": image_ids,
    }


def build_dataloader(
    cfg: dict,
    split: str = "train",
    batch_size: int | None = None,
    num_workers: int | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    """构建 DataLoader。

    Args:
        cfg: 完整配置；读取 data 段与 train 段。
        split: "train" | "val"。
        batch_size / num_workers / shuffle：None 时从 cfg 推断。
    """
    ds = build_coco_dataset(cfg, split=split)

    train_cfg: dict = cfg.get("train", {})
    if batch_size is None:
        batch_size = train_cfg.get("batch_size", 16)
    if num_workers is None:
        num_workers = train_cfg.get("num_workers", 4)
        if sys.platform == "win32":
            num_workers = 0  # Windows spawn 多进程可能死锁
    if shuffle is None:
        shuffle = (split == "train")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )