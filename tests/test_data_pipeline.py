"""数据管道烟雾测试（M1 第二步 a）。

合成一份 mini-COCO fixture（3 张小图 + instances json，使用项目 class_map 内的真实 COCO
category_id 1/2/3/25/28/39/62），验证：
    - Dataset 实例化、7 类筛选 + ID 重映射正确
    - transforms.v2 + bbox 联动不报错，输出 image shape 与 input_size 一致
    - DataLoader + collate_fn 能拼出 [B,3,H,W] 与变长 boxes/labels list
    - 跨平台 pathlib 生效

不依赖真实 COCO 下载；不依赖 GPU。
"""

import json
import sys
from pathlib import Path

# pytest 可选：脚本模式下也可直接 `python tests/test_data_pipeline.py` 运行 __main__
try:
    import pytest
except ImportError:
    pytest = None

# pytest 缺失时把 fixture 装饰器降级为 no-op，便于脚本模式直接运行
_fixture = pytest.fixture(scope="module") if pytest is not None else (lambda f: f)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image


# 真实 COCO category_id（与 configs/default.yaml 对应，见 coco_id_to_internal）
CATS = [
    {"id": 1,  "name": "person",   "supercategory": "none"},
    {"id": 2,  "name": "bicycle",  "supercategory": "none"},
    {"id": 3,  "name": "car",      "supercategory": "none"},
    {"id": 25, "name": "backpack", "supercategory": "none"},
    {"id": 28, "name": "suitcase", "supercategory": "none"},
    {"id": 39, "name": "bottle",   "supercategory": "none"},
    {"id": 62, "name": "chair",    "supercategory": "none"},
]

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "mini_coco"


def _make_fixture():
    """生成/确保存在 mini-COCO fixture，返回 (root, train_split, val_split)。"""
    root = FIXTURE_ROOT
    train_dir = root / "train2017"
    val_dir = root / "val2017"
    ann_dir = root / "annotations"
    for d in (train_dir, val_dir, ann_dir):
        d.mkdir(parents=True, exist_ok=True)

    def make_split(split_dir: Path, split_name: str):
        images = []
        anns = []
        ann_id = 1
        for i in range(1, 4):  # 3 张 96x64 图
            im = Image.new("RGB", (96, 64), color=(40 + i * 20, 80, 200 - i * 30))
            fname = f"{i:012d}.jpg"
            im.save(split_dir / fname, quality=80)
            images.append({"id": i, "file_name": fname, "height": 64, "width": 96})
            # 每张图 2 个框，类别覆盖部分
            cats_for_img = [CATS[0], CATS[1], CATS[2], CATS[6]][i - 1:i + 1]
            for j, c in enumerate(cats_for_img):
                x = 4 + j * 30
                y = 4 + j * 10
                w = 20
                h = 24
                anns.append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": c["id"],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1
        json_obj = {
            "info": {"description": "LEAD-Net mini COCO fixture"},
            "licenses": [],
            "images": images,
            "annotations": anns,
            "categories": CATS,
        }
        with open(ann_dir / f"instances_{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(json_obj, f)

    make_split(train_dir, "train2017")
    make_split(val_dir, "val2017")
    return root, "train2017", "val2017"

@_fixture
def mini_cfg():
    root, _, _ = _make_fixture()
    from lead_net.utils import load_config
    cfg = load_config("configs/baseline_ssd.yaml")
    cfg = resolve_paths_in(cfg)
    cfg["paths"]["dataset_root"] = root
    cfg["data"]["input_size"] = 64  # 用小尺寸加快测试
    cfg["data"]["train_split"] = "train2017"
    cfg["data"]["val_split"] = "val2017"
    return cfg


def test_fixture_classes():
    root, _, _ = _make_fixture()
    assert (root / "annotations" / "instances_train2017.json").exists()
    assert (root / "train2017").is_dir()


def test_dataset_filter_and_remap(mini_cfg):
    from lead_net.data import build_coco_dataset
    ds = build_coco_dataset(mini_cfg, split="train")
    assert len(ds) > 0
    sample = ds[0]
    # image: [3,64,64] float (经 to_tensor + normalize)
    assert sample["image"].shape[-2:] == (64, 64)
    assert sample["image"].dtype.is_floating_point
    # labels 全部落在 0..6
    lbls = sample["labels"].tolist()
    assert all(0 <= l <= 6 for l in lbls), lbls
    # 项目内部类别与 class_map 一一对应：用 person->0, bicycle->5 等
    assert set(lbls).issubset(set(mini_cfg["class_map"].keys()))


def test_transforms_keep_bbox_consistent(mini_cfg):
    from lead_net.data import build_coco_dataset
    ds = build_coco_dataset(mini_cfg, split="train")
    sample = ds[0]
    # boxes 数量与 labels 一致
    assert sample["boxes"].shape[0] == sample["labels"].shape[0]


def test_dataloader_collate(mini_cfg):
    from lead_net.data import build_dataloader
    loader = build_dataloader(mini_cfg, split="train", batch_size=2, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    assert batch["image"].ndim == 4 and batch["image"].shape[1] == 3
    assert isinstance(batch["boxes"], list)
    assert isinstance(batch["labels"], list)
    assert batch["image"].shape[0] == len(batch["boxes"])


def test_val_transforms(mini_cfg):
    from lead_net.data import build_coco_dataset
    ds = build_coco_dataset(mini_cfg, split="val")
    sample = ds[1]
    assert sample["image"].shape[-2:] == (64, 64)


if __name__ == "__main__":
    # 不带 pytest 也能手动验证
    _make_fixture()
    from lead_net.utils import load_config, resolve_paths_in
    cfg = load_config("configs/baseline_ssd.yaml")
    cfg = resolve_paths_in(cfg)
    cfg["paths"]["dataset_root"] = FIXTURE_ROOT
    cfg["data"]["input_size"] = 64
    cfg["data"]["train_split"] = "train2017"
    cfg["data"]["val_split"] = "val2017"
    from lead_net.data import build_dataloader
    loader = build_dataloader(cfg, split="train", batch_size=2, num_workers=0, shuffle=False)
    b = next(iter(loader))
    print("[ok] pipeline smoke: image", tuple(b["image"].shape),
          "| boxes lens=", [int(x.shape[0]) for x in b["boxes"]],
          "| labels=", [x.tolist() for x in b["labels"]])