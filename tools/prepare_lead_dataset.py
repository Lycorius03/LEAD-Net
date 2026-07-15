#!/usr/bin/env python
"""LEAD-Net Dataset Preparation Script.

This script builds a custom class-balanced dataset (YOLO format) for fine-tuning
LEAD-Net on dynamic obstacle detection. It performs:
  1. Class-balanced sampling of COCO 2017 (7 categories: person, bicycle, car, backpack, suitcase, chair, bottle).
  2. Conversion of COCO annotations (JSON) to YOLO format (.txt).
  3. Conversion of a subset of KITTI (road scene) annotations to YOLO format, mapped consistently.
  4. Assembly of a unified dataset in `data/lead_subset/` containing:
     - train: ~17.5k COCO + 2.5k KITTI (Total ~20k images)
     - val: ~2.2k COCO val images containing the 7 target categories
     - test: ~4.9k remaining KITTI images for out-of-domain generalization testing
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image

# ─── Configuration ───────────────────────────────────────
CLASS_MAP = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "backpack": 3,
    "suitcase": 4,
    "chair": 5,
    "bottle": 6
}

# Mapping COCO names to our internal class names/IDs
COCO_NAME_MAP = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "backpack": "backpack",
    "suitcase": "suitcase",
    "chair": "chair",
    "bottle": "bottle"
}

# Mapping KITTI names to our first 3 internal IDs
KITTI_MAP = {
    "Pedestrian": 0, "pedestrian": 0,
    "Cyclist": 1, "cyclist": 1,
    "Car": 2, "car": 2,
    "Van": 2, "Truck": 2, "Person_sitting": 0, "Person-Sitting": 0
}

# Class-balanced target IMAGE counts for COCO train subset
# 计数单位是"包含该类的图片数"，不是目标实例数。
# 每类 ~3000 张图片，7 类 × 3000 ≈ 21000，但多类共存图片去重后约 15000~18000。
COCO_TRAIN_LIMITS = {
    "person": 3500,
    "car": 3500,
    "bicycle": 2500,
    "chair": 2500,
    "bottle": 2500,
    "backpack": 2000,
    "suitcase": 1500
}

# ─── Paths ───────────────────────────────────────────────
COCO_ROOT = Path("data/coco")
COCO_TRAIN_IMG_DIR = COCO_ROOT / "train2017"
COCO_VAL_IMG_DIR = COCO_ROOT / "val2017"
COCO_ANN_DIR = COCO_ROOT / "annotations_trainval2017" / "annotations"

KITTI_ROOT = Path("data/KITTI")
KITTI_IMG_DIR = KITTI_ROOT / "data_object_image_2" / "training" / "image_2"
KITTI_LBL_DIR = KITTI_ROOT / "data_object_label_2" / "training" / "label_2"

OUT_ROOT = Path("data/lead_subset")


def setup_dirs():
    """Create output directories, removing old if existed."""
    if OUT_ROOT.exists():
        print(f"[setup] Removing existing output folder: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)
    
    for split in ["train", "val", "test"]:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    print("[setup] Output directories initialized.")


def coco_to_yolo(bbox, img_w, img_h):
    """Convert COCO bbox [x_min, y_min, w, h] to YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def kitti_to_yolo(bbox_left, bbox_top, bbox_right, bbox_bottom, img_w, img_h):
    """Convert KITTI bbox [left, top, right, bottom] to YOLO [cx, cy, w, h] normalized."""
    w = bbox_right - bbox_left
    h = bbox_bottom - bbox_top
    cx = bbox_left + w / 2
    cy = bbox_top + h / 2
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def process_coco_split(ann_file: Path, img_src_dir: Path, split: str, is_train: bool):
    """Filter COCO annotations, balance classes, and output in YOLO format."""
    print(f"[coco] Loading annotations from {ann_file.name}...")
    with open(ann_file) as f:
        coco = json.load(f)

    # Build category lookup
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    target_cat_ids = {c["id"]: c["name"] for c in coco["categories"] if c["name"] in COCO_NAME_MAP}
    
    # Map image ID to info
    img_id_to_info = {img["id"]: img for img in coco["images"]}
    
    # Group annotations by image
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in target_cat_ids:
            anns_by_img[ann["image_id"]].append(ann)

    print(f"[coco] Found {len(anns_by_img)} images containing target categories in {split}.")

    selected_img_ids = set()
    
    if is_train:
        # Class-balanced greedy sampling
        random.seed(42)
        selected_counts = {name: 0 for name in COCO_TRAIN_LIMITS}
        
        # Group image IDs by their target categories
        cat_to_imgs = defaultdict(list)
        for img_id, anns in anns_by_img.items():
            img_cats = {target_cat_ids[ann["category_id"]] for ann in anns}
            for cat in img_cats:
                cat_to_imgs[cat].append(img_id)

        # Shuffle lists for reproducibility
        for cat in cat_to_imgs:
            random.shuffle(cat_to_imgs[cat])

        # Sort categories rarest-first to prioritize them
        sorted_cats = sorted(COCO_TRAIN_LIMITS.keys(), key=lambda name: COCO_TRAIN_LIMITS[name])
        
        for cat in sorted_cats:
            limit = COCO_TRAIN_LIMITS[cat]
            img_list = cat_to_imgs[cat]
            for img_id in img_list:
                if selected_counts[cat] >= limit:
                    break
                if img_id in selected_img_ids:
                    continue
                
                # Check if this image has valid dimensions in json
                img_info = img_id_to_info[img_id]
                if not img_info.get("file_name"):
                    continue

                selected_img_ids.add(img_id)
                # 每张图每类只计 1 次（图片级计数，非实例级）
                img_cats = {target_cat_ids[ann["category_id"]] for ann in anns_by_img[img_id]}
                for c_name in img_cats:
                    selected_counts[c_name] += 1
        
        print("[coco] Selected class-balanced image counts:")
        for cat, cnt in selected_counts.items():
            print(f"  - {cat}: {cnt} images containing this class (target limit: {COCO_TRAIN_LIMITS[cat]})")
    else:
        # Validation set: keep all valid images containing target categories (no sampling)
        for img_id in anns_by_img:
            if img_id_to_info[img_id].get("file_name"):
                selected_img_ids.add(img_id)

    print(f"[coco] Total selected images for {split}: {len(selected_img_ids)}")

    # Copy images and write labels
    img_dest_dir = OUT_ROOT / "images" / split
    lbl_dest_dir = OUT_ROOT / "labels" / split
    
    copied_count = 0
    for img_id in selected_img_ids:
        img_info = img_id_to_info[img_id]
        filename = img_info["file_name"]
        src_path = img_src_dir / filename
        
        if not src_path.exists():
            continue
            
        # Copy image
        shutil.copy(src_path, img_dest_dir / filename)
        copied_count += 1
        
        # Write YOLO labels
        img_w = img_info["width"]
        img_h = img_info["height"]
        
        label_file = lbl_dest_dir / f"{Path(filename).stem}.txt"
        with open(label_file, "w") as lf:
            for ann in anns_by_img[img_id]:
                cat_name = target_cat_ids[ann["category_id"]]
                class_id = CLASS_MAP[cat_name]
                cx, cy, w, h = coco_to_yolo(ann["bbox"], img_w, img_h)
                lf.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                
    print(f"[coco] Successfully processed & copied {copied_count} images to {split}.")


def process_kitti_split():
    """Convert and split KITTI dataset (2500 train, rest test)."""
    print("[kitti] Scanning KITTI images...")
    
    # Get all valid KITTI label txt files
    labels = sorted(list(KITTI_LBL_DIR.glob("*.txt")))
    kitti_samples = []
    
    for lbl in labels:
        img = None
        for ext in (".png", ".jpg", ".jpeg"):
            c = KITTI_IMG_DIR / f"{lbl.stem}{ext}"
            if c.is_file():
                img = c
                break
        if img:
            kitti_samples.append((img, lbl))

    print(f"[kitti] Found {len(kitti_samples)} matching image-label pairs.")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(kitti_samples)
    
    train_samples = kitti_samples[:2500]
    test_samples = kitti_samples[2500:]
    
    print(f"[kitti] Train subset size: {len(train_samples)}")
    print(f"[kitti] Test subset size: {len(test_samples)}")
    
    def copy_and_convert(samples, split):
        img_dest = OUT_ROOT / "images" / split
        lbl_dest = OUT_ROOT / "labels" / split
        
        count = 0
        for img_path, lbl_path in samples:
            # 1. Copy image
            shutil.copy(img_path, img_dest / img_path.name)
            
            # 2. Get image size
            with Image.open(img_path) as im:
                img_w, img_h = im.size
                
            # 3. Read and convert label
            yolo_lines = []
            with open(lbl_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    kitti_type = parts[0]
                    if kitti_type in KITTI_MAP:
                        class_id = KITTI_MAP[kitti_type]
                        # bounding box left, top, right, bottom
                        left, top, right, bottom = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                        cx, cy, w, h = kitti_to_yolo(left, top, right, bottom, img_w, img_h)
                        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                        
            # 4. Write YOLO label
            with open(lbl_dest / f"{lbl_path.stem}.txt", "w") as lf:
                lf.writelines(yolo_lines)
            count += 1
            
        print(f"[kitti] Processed {count} KITTI images to {split}.")

    copy_and_convert(train_samples, "train")
    copy_and_convert(test_samples, "test")


def main():
    setup_dirs()
    
    # 1. Process COCO Train -> data/lead_subset/train
    process_coco_split(
        ann_file=COCO_ANN_DIR / "instances_train2017.json",
        img_src_dir=COCO_TRAIN_IMG_DIR,
        split="train",
        is_train=True
    )
    
    # 2. Process COCO Val -> data/lead_subset/val
    process_coco_split(
        ann_file=COCO_ANN_DIR / "instances_val2017.json",
        img_src_dir=COCO_VAL_IMG_DIR,
        split="val",
        is_train=False
    )
    
    # 3. Process KITTI -> data/lead_subset/train and data/lead_subset/test
    process_kitti_split()
    
    print("\n=== Dataset Preparation Complete ===")
    print(f"Unified dataset generated at: {OUT_ROOT.resolve()}")
    print(f"  Train images: {len(list((OUT_ROOT / 'images' / 'train').glob('*')))}")
    print(f"  Val images:   {len(list((OUT_ROOT / 'images' / 'val').glob('*')))}")
    print(f"  Test images:  {len(list((OUT_ROOT / 'images' / 'test').glob('*')))}")


if __name__ == "__main__":
    main()
