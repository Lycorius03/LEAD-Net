"""检测 transforms 工厂（torchvision transforms.v2，支持 bbox 联动）。

依据：
    - docs/DATASET.md §数据预处理规则：分辨率/归一化/增强策略（M1 初值待调）。
    - docs/ARCHITECTURE.md §配置驱动：参数从 cfg 读取。
    - docs/MODULES.md §5：保存具体预处理参数供可复现性回填。

设计细节（M1 第二步 a，理由见 CHANGELOG）：
    - 使用 torchvision.transforms.v2，自动处理 BoundingBoxes 与 Image 联动。
    - Resize 到 input_size（保持长宽比 -> 直接 resize 到正方形，SSD 简化做法，待 M3 锚框时再决定是否 letterbox）。
    - 归一化：ImageNet mean/std（cfg.data.mean / cfg.data.std）。
    - train 增强：RandomHorizontalFlip（联动 boxes）→ ColorJitter（仅图像）→ Resize → ToDtype float → Normalize。
    - val：Resize → ToDtype float → Normalize（不随机增强）。
    - 所有 transform 在 cfg.data.{train,val}_transforms 列出的名称中按序生成；不在白名单的名称忽略以保安全。

白名单（避免误用危险/破坏 bbox 的增强）：
    resize, random_horizontal_flip, color_jitter, to_tensor, normalize
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import Image as TVImage, BoundingBoxes


_BUILDERS = {
    "resize": lambda cfg: T.Resize(size=(cfg["data"]["input_size"], cfg["data"]["input_size"]), antialias=True),
    "random_horizontal_flip": lambda cfg: T.RandomHorizontalFlip(p=1.0),
    "color_jitter": lambda cfg: T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
    "to_tensor": lambda cfg: _ToDtypeFloat(),
    "normalize": lambda cfg: T.Normalize(mean=cfg["data"]["mean"], std=cfg["data"]["std"], inplace=False),
}

# 训练默认顺序，避免顺序错乱影响 bbox（先几何后像素强度）
_DEFAULT_TRAIN = ["random_horizontal_flip", "color_jitter", "resize", "to_tensor", "normalize"]
_DEFAULT_VAL = ["resize", "to_tensor", "normalize"]


class _ToDtypeFloat(T.Transform):
    """把 image 转 float32 而 boxes 保持 float。

    transforms.v2 的 ToDtype 对 image 用 float32、bboxes 不变；此处显式包装便于命名一致。
    """

    def __init__(self):
        super().__init__()
        self._inner = T.ToDtype({TVImage: torch.float32, BoundingBoxes: torch.float32}, scale=False)

    def forward(self, sample: dict) -> dict:
        out = dict(sample)
        out["image"] = self._inner(sample["image"])
        # boxes 保持 dtype 不变（已是 float），labels 保持 long
        return out


def build_transforms(cfg: dict, split: str = "train") -> T.Transform:
    """构建 transforms.v2 Compose。

    Args:
        cfg: 完整配置；读取 cfg["data"] 段。
        split: "train" | "val"。

    Returns:
        torchvision.transforms.v2.Compose（input=dict(sample) -> dict(sample)）。
    """
    data_cfg: dict = cfg.get("data", {})
    if split == "train":
        keys = "train_transforms"
        default = _DEFAULT_TRAIN
    else:
        keys = "val_transforms"
        default = _DEFAULT_VAL
    names = list(data_cfg.get(keys, default) or default)

    transforms = []
    for name in names:
        builder = _BUILDERS.get(name)
        if builder is None:
            # 未在白名单的忽略，避免引入未知的破坏性 transform
            continue
        transforms.append(builder(cfg))
    if not transforms:
        transforms = [_ToDtypeFloat()]
    return T.Compose(transforms)