"""TXT 格式检测数据集（YOLO 标注格式兼容）。

支持的标注格式（归一化 cxcywh）：
    每张图片对应一个同名的 .txt 文件，每行：
        class_id cx cy w h
    其中 cx,cy,w,h 均归一化到 [0,1]（相对于原图宽高）。

数据集目录结构（推荐）：
    <root>/
        images/
            train/   (所有训练图片)
            val/     (所有验证图片)
        labels/
            train/   (所有训练标注 .txt)
            val/     (所有验证标注 .txt)

返回格式（与 LEADCOCODetection 兼容）：
    dict(image=tv_tensor[C,H,W]float, boxes=BoundingBoxes[N,4] XYWH abs,
         labels=LongTensor[N], image_id=LongTensor[1])
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image as TVImage

from .transforms import build_transforms


def build_txt_dataset(
    cfg: dict,
    split: str = "train",
    transforms: Any = None,
) -> "TXDetection":
    """构建 TXT 格式检测 Dataset（归一化 cxcywh 标注）。

    Args:
        cfg: 完整配置；读取 paths.dataset_root / data.train_image_dir /
             data.val_image_dir / data.label_dir / class_map / num_classes。
        split: "train" | "val"。
        transforms: torchvision.transforms.v2.Compose；None 时用 build_transforms(cfg, split)。
    """
    data_cfg: dict = cfg.get("data", {})
    dataset_root: Path = Path(str(cfg["paths"]["dataset_root"]))

    images_dir = dataset_root / data_cfg.get(
        f"{split}_image_dir", f"images/{split}"
    )
    labels_dir = dataset_root / data_cfg.get(
        "label_dir", "labels"
    ) / split

    num_classes: int = cfg.get("num_classes", 3)
    mosaic_prob: float = data_cfg.get("mosaic", 0.0) if split == "train" else 0.0
    input_size: int = data_cfg.get("input_size", 320)

    if transforms is None:
        transforms = build_transforms(cfg, split=split)

    return TXDetection(
        images_dir=images_dir,
        labels_dir=labels_dir,
        num_classes=num_classes,
        transforms=transforms,
        split=split,
        mosaic_prob=mosaic_prob,
        input_size=input_size,
    )


class TXDetection(Dataset):
    """TXT 格式检测数据集（归一化 cxcywh 标注）。

    返回与 LEADCOCODetection 相同结构的数据样本，可直接对接
    现有的 DataLoader / collate_fn / transforms / Trainer 管线。
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        num_classes: int = 3,
        transforms: Any = None,
        split: str = "train",
        mosaic_prob: float = 0.0,
        input_size: int = 320,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.num_classes = num_classes
        self._transforms = transforms
        self.split = split
        self.mosaic_prob = mosaic_prob if split == "train" else 0.0
        self.input_size = input_size

        # 收集所有图片路径（按文件名排序保证可复现）
        self._image_paths: list[Path] = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"图像目录不存在: {self.images_dir}.\n"
                f"请先运行 tools/prepare_lead_dataset.py 转换数据集，"
                f"或确认 dataset_root 配置正确。"
            )

        # 过滤掉没有对应标注文件的图片
        valid = []
        missing = 0
        for img_path in self._image_paths:
            label_path = self._label_path(img_path)
            if label_path.is_file():
                valid.append(img_path)
            else:
                missing += 1

        self._image_paths = valid
        if missing > 0:
            print(f"[info] {self.split}: {missing} images have no label file, "
                  f"{len(self._image_paths)} valid")

        if len(self._image_paths) == 0:
            print(f"[warn] {self.split}: 无可用图片！请确认 {self.images_dir} "
                  f"包含图片且 {self.labels_dir} 有对应 .txt 标注文件。")

    def _label_path(self, img_path: Path) -> Path:
        """获取图片对应的标注文件路径。"""
        return self.labels_dir / f"{img_path.stem}.txt"

    def _parse_labels(self, label_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """解析 TXT 标注文件（归一化 cxcywh），返回 (boxes_xywh_pixel, labels)。

        boxes 为绝对像素坐标 [x, y, w, h]（xywh，左上角+宽高）。
        labels 为类别 ID（0-indexed）。
        """
        boxes: list[list[float]] = []
        labels: list[int] = []
        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            if w <= 0 or h <= 0:
                continue
            if cls_id < 0 or cls_id >= self.num_classes:
                continue
            boxes.append([cx, cy, w, h])
            labels.append(cls_id)

        return (
            torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
        )

    @staticmethod
    def _cxcywh_to_xywh_pixel(
        boxes: torch.Tensor, img_w: int, img_h: int
    ) -> torch.Tensor:
        """归一化 cxcywh → 绝对像素 xywh（左上角+宽高）。"""
        if boxes.numel() == 0:
            return boxes
        cx = boxes[:, 0] * img_w
        cy = boxes[:, 1] * img_h
        w = boxes[:, 2] * img_w
        h = boxes[:, 3] * img_h
        x = cx - w / 2
        y = cy - h / 2
        return torch.stack([x, y, w, h], dim=-1)

    def __len__(self) -> int:
        return len(self._image_paths)

    def _load_sample(self, img_idx: int) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
        """加载单张图片及其标注（原始坐标）。"""
        img_path = self._image_paths[img_idx]
        label_path = self._label_path(img_path)
        image = Image.open(img_path).convert("RGB")
        boxes_cxcywh, labels = self._parse_labels(label_path)
        return image, boxes_cxcywh, labels

    def _mosaic(self, idx: int) -> dict[str, Any]:
        """Mosaic 数据增强：4 张图拼成 1 张。

        参考 Mosaic 增强实现：随机选取 3 张额外图片，在 input_size×input_size
        画布上按随机中心点划分为 4 个象限，每张图放入一个象限并调整 bbox 偏移。
        """
        s = self.input_size
        # 随机中心点（抖动 ±25%）
        xc = int(random.uniform(s * 0.25, s * 0.75))
        yc = int(random.uniform(s * 0.25, s * 0.75))

        # 选 3 个额外索引
        n = len(self)
        indices = [idx] + random.choices(range(n), k=3)
        random.shuffle(indices)

        mosaic_img = Image.new("RGB", (s, s), (114, 114, 114))
        all_boxes_xywh: list[list[float]] = []
        all_labels: list[int] = []
        final_img_w, final_img_h = s, s

        # 4 个象限：(x1, y1, x2, y2) 画布像素坐标
        quads = [
            (0, 0, xc, yc),            # top-left
            (xc, 0, s, yc),            # top-right
            (0, yc, xc, s),            # bottom-left
            (xc, yc, s, s),              # bottom-right
        ]

        for i, (img_idx, (qx1, qy1, qx2, qy2)) in enumerate(zip(indices, quads)):
            img, boxes_cxcywh, labels = self._load_sample(img_idx)
            if boxes_cxcywh.numel() == 0:
                continue

            iw, ih = img.size
            qw = qx2 - qx1
            qh = qy2 - qy1

            # 缩放图片填满象限（保持长宽比可能导致不完全填充，用拉伸更简单）
            if iw > 0 and ih > 0:
                img = img.resize((qw, qh), Image.BILINEAR)

            mosaic_img.paste(img, (qx1, qy1))

            # 转换 bbox：归一化 cxcywh → 画布绝对 xywh
            for j in range(boxes_cxcywh.shape[0]):
                cx, cy, w, h = boxes_cxcywh[j].tolist()
                # 原图绝对坐标
                abs_cx = cx * iw
                abs_cy = cy * ih
                abs_w = w * iw
                abs_h = h * ih
                # 缩放到象限尺寸
                scale_x = qw / iw
                scale_y = qh / ih
                new_w = abs_w * scale_x
                new_h = abs_h * scale_y
                new_x = abs_cx * scale_x - new_w / 2 + qx1
                new_y = abs_cy * scale_y - new_h / 2 + qy1
                # 裁剪到画布内
                new_x = max(0, new_x)
                new_y = max(0, new_y)
                new_w = min(new_w, s - new_x)
                new_h = min(new_h, s - new_y)
                if new_w <= 1 or new_h <= 1:
                    continue
                all_boxes_xywh.append([new_x, new_y, new_w, new_h])
                all_labels.append(int(labels[j].item()))

        if not all_boxes_xywh:
            # Mosaic 后无有效框 → 返回空标注。
            # 原因：伪造的 [[0,0,1,1]] 永远不会匹配任何 anchor (IoU<0.5)，
            # 但空列表让下游明确知道"此图无目标"。
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.long)
            tv_img = TVImage(mosaic_img)
            boxes_bb = BoundingBoxes(
                boxes_t,
                format=BoundingBoxFormat.XYWH,
                canvas_size=(s, s),
            )
            sample = {
                "image": tv_img,
                "boxes": boxes_bb,
                "labels": labels_t,
                "image_id": torch.tensor([idx], dtype=torch.long),
            }
            if self._transforms is not None:
                sample = self._transforms(sample)
            return sample

        boxes_t = torch.tensor(all_boxes_xywh, dtype=torch.float32)
        labels_t = torch.tensor(all_labels, dtype=torch.long)

        tv_img = TVImage(mosaic_img)
        boxes_bb = BoundingBoxes(
            boxes_t,
            format=BoundingBoxFormat.XYWH,
            canvas_size=(s, s),
        )

        sample = {
            "image": tv_img,
            "boxes": boxes_bb,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.long),
        }

        if self._transforms is not None:
            sample = self._transforms(sample)
        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Mosaic 增强（仅训练时，概率 mosaic_prob）
        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            return self._mosaic(idx)

        img_path = self._image_paths[idx]
        label_path = self._label_path(img_path)

        image = Image.open(img_path).convert("RGB")
        w_img, h_img = image.size

        boxes_cxcywh, labels = self._parse_labels(label_path)
        boxes_xyxy = self._cxcywh_to_xywh_pixel(boxes_cxcywh, w_img, h_img)

        tv_img = TVImage(image)
        boxes_bb = BoundingBoxes(
            boxes_xyxy,
            format=BoundingBoxFormat.XYWH,
            canvas_size=(h_img, w_img),
        )

        sample = {
            "image": tv_img,
            "boxes": boxes_bb,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.long),
        }

        if self._transforms is not None:
            sample = self._transforms(sample)
        return sample
