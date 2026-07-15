"""DataLoader 工厂 + 检测 collate + GPU 加速批处理。

依据：
    - docs/MODULES.md §5 训练与量化 Pipeline：DataLoader 与训练流程相关。
    - docs/ARCHITECTURE.md §配置驱动：num_workers/batch_size 从 cfg["train"] 读。

设计细节（M1 第二步 a）：
    - Dataset.__getitem__ 返回 dict(image, boxes, labels, image_id, ...)，长度可变；
      用自定义 collate 把 image 堆为 [B,3,H,W]，boxes/labels 保持 list[Tensor]（变长）。
    - Windows 下 DataLoader num_workers!=0 可能触发问题；cfg 默认值已在 baseline_ssd.yaml 标注待调。

GPU 加速（--gpu-pipeline / data.gpu_pipeline）:
    - CPU workers：JPEG decode (pillow-simd) + augment + resize → CPU float tensor
    - collate 堆叠后立刻 pin→GPU，normalize 在 GPU 上完成
    - CPU 只做解码+增强，GPU 做 normalize，PCIe 传输与下一批解码重叠
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

from .coco_dataset import build_coco_dataset
from .txt_dataset import build_txt_dataset


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


class GPUBatchProcessor:
    """Post-collate GPU 处理：传输 + normalize 在 GPU 上完成。

    CPU workers 做完 decode+augment+resize 后产出 float tensor，
    collate 堆叠 → GPUBatchProcessor 把 batch 送上 GPU 并 normalize。
    这样 CPU 可以立刻投入下一批，PCIe 传输和 decode 重叠。
    """

    def __init__(self, mean: list[float], std: list[float], device: str = "cuda"):
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.device = device

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        images = batch["image"].to(self.device, non_blocking=True)
        # Normalize on GPU (image is already float [0,1] from ToDtype)
        images = (images - self.mean) / self.std
        batch["image"] = images
        return batch


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

    Returns:
        DataLoader；若 cfg.data.gpu_pipeline=true，可通过
        dataloader.gpu_processor 获取 GPUBatchProcessor 实例，
        在训练循环中对每个 batch 调用 processor(batch)。
    """
    data_cfg: dict = cfg.get("data", {})
    dataset_type: str = data_cfg.get("dataset_type", "coco")

    if dataset_type == "txt":
        ds = build_txt_dataset(cfg, split=split)
    else:
        ds = build_coco_dataset(cfg, split=split)

    train_cfg: dict = cfg.get("training") or cfg.get("train", {})
    if batch_size is None:
        batch_size = train_cfg.get("batch_size", 16)
    if num_workers is None:
        num_workers = train_cfg.get("num_workers", 4)
        if sys.platform == "win32":
            num_workers = 0  # Windows spawn 多进程可能死锁
    if shuffle is None:
        shuffle = (split == "train")

    prefetch = train_cfg.get("prefetch_factor", 4) if num_workers > 0 else None

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch,
    )

    # Attach GPU processor if enabled (training only; val keeps CPU normalize)
    use_gpu = data_cfg.get("gpu_pipeline", False) and split == "train"
    if use_gpu and torch.cuda.is_available():
        dl.gpu_processor = GPUBatchProcessor(
            mean=data_cfg.get("mean", [0.485, 0.456, 0.406]),
            std=data_cfg.get("std", [0.229, 0.224, 0.225]),
        )
    else:
        dl.gpu_processor = None

    return dl