"""COCO 检测数据集（基于 torchvision.datasets.CocoDetection 派生，本类别子集）。

依据：
    - docs/DATASET.md：主数据集 COCO 2017；7 类子集（初值）。
    - docs/ARCHITECTURE.md §跨平台：路径用 pathlib.Path。
    - docs/MODULES.md §3 Detection Head：类别数与筛选由 DATASET.md 决定。

设计细节（M1 第二步 a 回填，理由见 CHANGELOG）：
    - 类别人筛：COCO 标注按 category_id 过滤到项目 class_map 内的类别；
      通过 cfg["coco_id_to_internal"] 把 COCO 原始 ID 映射为项目内部连续 ID（0..6）。
    - bbox 格式：COCO 原生 [x, y, w, h]（绝对像素），统一封装为
      torchvision.tv_tensors.BoundingBoxes(format="XYWH", canvas_size=(H,W))，
      便于 transforms.v2 自动联动图像变换。
    - 返回每条样本：dict(image=tv_tensor[H,W,3]float, boxes=BoundingBoxes[N,4],
      labels=LongTensor[N]（项目内部 ID），其它字段（area/iscrowd/image_id）保留供 loss 评估。
    - num_classes（不含背景类）：来自 cfg["num_classes"]。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.datasets import CocoDetection
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from .transforms import build_transforms


def build_coco_dataset(
    cfg: dict,
    split: str = "train",
    transforms: Any = None,
) -> "LEADCOCODetection":
    """构建 COCO 子集 Dataset。

    Args:
        cfg: 完整配置；读取 paths.dataset_root / class_map / coco_id_to_internal /
             data.train_split / data.val_split / data.use_class_subset / num_classes。
        split: "train" | "val"。
        transforms: torchvision.transforms.v2.Compose；若 None 则用 build_transforms(cfg, split)。
    """
    data_cfg: dict = cfg.get("data", {})
    split_name = data_cfg.get(
        f"{split}_split", "train2017" if split == "train" else "val2017"
    )

    dataset_root: Path = Path(str(cfg["paths"]["dataset_root"]))
    images_dir = dataset_root / split_name
    ann_file = dataset_root / "annotations" / f"instances_{split_name}.json"

    class_map: dict = cfg.get("class_map", {})
    coco_id_to_internal: dict = cfg.get("coco_id_to_internal", {})
    use_class_subset: bool = data_cfg.get("use_class_subset", True)

    if transforms is None:
        transforms = build_transforms(cfg, split=split)

    return LEADCOCODetection(
        images_dir=images_dir,
        ann_file=ann_file,
        class_map=class_map,
        coco_id_to_internal=coco_id_to_internal,
        use_class_subset=use_class_subset,
        transforms=transforms,
        split=split,
    )


class LEADCOCODetection(CocoDetection):
    """LEAD-Net COCO 检测子集。

    继承 torchvision CocoDetection 以复用 COCO json 加载与图片读取；
    重写 __getitem__：
        1) 过滤到目标 category_id 集合（若 use_class_subset）
        2) 把 COCO category_id 重映射为项目内部连续 ID
        3) 包装 image/boxes 为 tv_tensors，交给 transforms.v2 联动处理
    """

    def __init__(
        self,
        images_dir: Path,
        ann_file: Path,
        class_map: dict,
        coco_id_to_internal: dict,
        use_class_subset: bool = True,
        transforms: Any = None,
        split: str = "train",
    ):
        # 父类 CocoDetection 的 transforms 接口是 (img, target) 元组；我们的 transforms 是 dict 风格。
        # 为避免接口冲突，父类 transforms=None，子类自己挂 _leadnet_transforms 应用到 dict。
        super().__init__(root=str(images_dir), annFile=str(ann_file), transforms=None)
        self.images_dir = Path(images_dir)
        self.ann_file = Path(ann_file)
        self.class_map = dict(class_map)
        self.coco_id_to_internal = {int(k): int(v) for k, v in coco_id_to_internal.items()}
        self.use_class_subset = use_class_subset
        self._leadnet_transforms = transforms
        self.split = split

        # 仅保留在映射表内的目标 category；base 类不会过滤，需在 __getitem__ 中按样本过滤
        self._allowed_coco_ids: set[int] = set(self.coco_id_to_internal.keys())

        # 验证图像目录存在（跨平台：Linux/Win 克隆后目录可能不存在）
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"图像目录不存在: {self.images_dir}. "
                f"请确认 dataset_root 包含 '{self.images_dir.name}' 子目录，"
                f"或运行 tools/download_coco.py 下载数据集。"
            )

        # 过滤：只保留本地存在的图片（容错不完整下载）
        self._filter_existing_images()

    def _filter_existing_images(self) -> None:
        """移除 ids 中本地不存在的图片（容忍不完整下载）。"""
        original = len(self.ids)
        valid_ids = []
        missing = 0
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)
            if img_info:
                fname = img_info[0]["file_name"]
                if (self.images_dir / fname).exists():
                    valid_ids.append(img_id)
                else:
                    missing += 1
            else:
                missing += 1
        self.ids = valid_ids
        if missing > 0:
            print(f"[info] {self.split}: filtered {missing}/{original} missing images, "
                  f"{len(self.ids)} remaining")
        if len(self.ids) == 0 and original > 0:
            print(f"[warn] {self.split}: 无可用图片！请确认 {self.images_dir} 中包含"
                  f" COCO 图片文件，且 annotation JSON 中的文件名与之匹配。")

    def _filter_remap(self, anns: list[dict]) -> tuple[list[list[float]], list[int]]:
        """过滤+重映射一组标注，返回 [boxes_xywh], [internal_labels]。"""
        boxes: list[list[float]] = []
        labels: list[int] = []
        for a in anns:
            cid = int(a["category_id"])
            if self.use_class_subset and cid not in self._allowed_coco_ids:
                continue
            # 跳过 crowd（loss 不消费 crowd；评估时另行处理）
            if a.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            if self.use_class_subset:
                internal = self.coco_id_to_internal[cid]
            else:
                # 未筛类时也尽量做映射；不在表内者跳过保持可重复
                internal = self.coco_id_to_internal.get(cid)
                if internal is None:
                    continue
            boxes.append([float(x), float(y), float(w), float(h)])
            labels.append(internal)
        return boxes, labels

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image, anns = super().__getitem__(idx)  # PIL.Image, list[dict]
        boxes_xywh, labels = self._filter_remap(anns)

        from PIL import Image
        if not isinstance(image, Image.Image):
            raise TypeError(f"unexpected image type: {type(image)}")

        w_img, h_img = image.size
        if len(boxes_xywh) == 0:
            # 保留空张量；transforms 仍应处理（v2 对空 boxes 安全）
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes_t = torch.tensor(boxes_xywh, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)

        from torchvision.tv_tensors import Image as TVImage
        tv_img = TVImage(image)
        boxes_bb = BoundingBoxes(
            boxes_t,
            format=BoundingBoxFormat.XYWH,
            canvas_size=(h_img, w_img),
        )

        sample = {
            "image": tv_img,
            "boxes": boxes_bb,
            "labels": labels_t,
            "image_id": torch.tensor([self.ids[idx]], dtype=torch.long),
        }
        if self._leadnet_transforms is not None:
            sample = self._leadnet_transforms(sample)
        return sample