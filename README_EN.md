# LEAD-Net

**L**ightweight **E**dge-aware **A**ttention **D**etection **Net**work

[中文](README.md) | **English**

> A real-time visual perception system for embedded devices — enabling a smart car to **see**, **track**, and **avoid** obstacles with a single RGB camera.

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.11-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/cuda-12.8-green)](https://developer.nvidia.com/cuda-toolkit)
[![Ultralytics](https://img.shields.io/badge/ultralytics-8.4-orange)](https://ultralytics.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

LEAD-Net is a lightweight computer vision system designed for resource-constrained embedded platforms. It performs **object detection → tracking → obstacle avoidance → motion planning** on a single RGB camera input, outputting a minimal UART protocol to an STM32 MCU for PID control.

**Target platform:** OpenMV H7 Plus (STM32H743, 480 MHz Cortex-M7, 1 MB SRAM, no NPU)
**Training platform:** Cloud GPU (RTX 5090 32GB) / Local dev (RTX 5060 8GB)
**Use cases:** Smart car competitions, autonomous driving education, embedded AI research

---

## Architecture

LEAD-Net uses a **teacher–student distillation** architecture to bridge the gap between high-accuracy training and MCU-deployable inference:

```text
YOLO11n + LCA(Neck)          Teacher model (proves LCA effectiveness, RQ1/RQ2)
       │
       ▼  Knowledge distillation
   ┌───┴───┐
  FOMO    ST-YOLOXn          Student candidates (MCU-deployable)
 (fast)   (accurate)
       │
       ▼  INT8 QAT → TFLite → OpenMV
 OpenMV H7 Plus              Edge deployment (RQ3/RQ5)
```

### Detection (Training-side)

| Component | Description |
|-----------|-------------|
| Backbone | YOLO11n (Ultralytics, COCO-pretrained `yolo11n.pt`), 2.6M params, 6.5 GFLOPs |
| LCA | Lightweight Coordinate-aware Attention, injected at Neck P3, ~1.7K extra params |
| Loss | Varifocal + DFL + CIoU (Ultralytics native) |
| Assigner | TaskAlignedAssigner (Ultralytics native) |

### Decision & Control (Deploy-side, class-agnostic)

| Component | Description |
|-----------|-------------|
| Decision Engine | 5-layer filtering: confidence → ROI → distance → DL/CV fusion → risk |
| Behavior FSM | SEARCHING → TRACKING → OBSTACLE_DETECTED → AVOIDING → TARGET_REACQUIRE |
| NSA-KF Tracker | Noise-adaptive Kalman + DIOU matching, survives 3-5 frame occlusion |
| APF Avoider | Artificial potential field, outputs coordinate offset for STM32 |
| Speed Controller | 5-level bbox area → speed mapping, auto-decelerate near target |
| CV Fallback | Traditional vision backup: ground segmentation + blob detection |

### UART Output Protocol

```text
Detected:  "o{cx_offset},d{detected},a{area}\r\n"
Lost:      "o0,d0,a0\r\n"
```

| Field | Range | Meaning |
|-------|-------|---------|
| `o` | -160 ~ +160 | Target center offset from image center (negative=left, positive=right) |
| `d` | 0 / 1 | Target detected flag |
| `a` | pixels² | Target area (bbox w×h or heatmap blob size) |

---

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Teacher params | 2.6M | YOLO11n + LCA (~1.7K extra) |
| Teacher FLOPs | 6.5G | @640 input (training); @416 for fine-tuning |
| Input resolution | 416×416 | Cloud training; 320×320 local debug |
| Classes | 7 | person, bicycle, car, backpack, suitcase, chair, bottle |
| Tracking targets | ≤3 | Configurable |
| Occlusion tolerance | 3-5 frames | NSA-KF prediction bridging |
| mAP@0.5 | *pending cloud training* | Updated after RTX 5090 training completes |
| OpenMV FPS | *pending deployment* | Updated after INT8 TFLite deployment |

> Previous SSD-Lite+MobileNetV3 architecture achieved 8.5% mAP@0.5 after 180 epochs (architecture ceiling). YOLO11n baseline expected to reach 70%+ after fine-tuning on 7-class subset.

---

## Quick Start

### Prerequisites

- Python 3.11+ · PyTorch 2.11+ · CUDA 12.0+
- Training: ≥8 GB VRAM (local) / ≥32 GB (cloud)
- Inference: CPU or OpenMV H7 Plus

```bash
pip install -r requirements.txt
pip install ultralytics  # YOLO11 integration
```

### Local Smoke Test (5 minutes)

Validates the full pipeline (model loading + LCA injection + weight transfer + training + eval) on 100 sample images:

```bash
# Windows (torchenv)
$env:PYTHONPATH = "."
python tools/smoke_verify_models.py

# Linux
PYTHONPATH=. python tools/smoke_verify_models.py
```

Expected: `baseline: PASS` + `lca_r16: PASS`

### Cloud Full Training (RTX 5090 32GB)

```bash
# On AutoDL cloud instance
cd /root/autodl-tmp/LEAD-Net
pip install ultralytics pycocotools
bash scripts/train_cloud.sh smoke     # 1-epoch validation
bash scripts/train_cloud.sh           # 4 variants × 180 epochs
```

### Evaluation

```bash
# COCO mAP on validation set
PYTHONPATH=. python -c "
from ultralytics import YOLO
m = YOLO('outputs/cloud/runs/lca_r16/weights/best.pt')
metrics = m.val(data='configs/lead_subset_ultralytics.yaml', imgsz=416)
print(f'mAP@0.5: {metrics.box.map50:.4f}')
"

# Build hard-scene subsets (occlusion / small targets / red background)
python tools/build_hard_subsets.py
```

---

## Project Structure

```text
LEAD-Net/
├── lead_net/
│   ├── models/
│   │   ├── attention/           # LCA module (ultralytics-adapted + legacy SSD)
│   │   ├── yolo/                # YOLO11 integration
│   │   │   ├── lca_adapter.py       # Register LCA to ultralytics
│   │   │   ├── weight_remapper.py   # Fix weight transfer after LCA insertion
│   │   │   ├── data_adapter.py      # lead_subset → ultralytics data.yaml
│   │   │   └── yamls/               # 4 YAML configs (baseline + LCA r=8/16/32)
│   │   ├── backbone.py          # MobileNetV3 (legacy SSD path, retained)
│   │   ├── detection_head.py    # SSD-Lite (legacy, retained for ablation)
│   │   ├── loss.py              # MultiBoxLoss (legacy)
│   │   └── lead_net.py          # Model assembly (routes YOLO / SSD)
│   ├── data/                    # Dataset · Transforms · DataLoader
│   ├── engine/                  # Trainer · Evaluator · Scheduler
│   ├── tracking/                # NSA-KF · DIOU · MOSSE
│   ├── decision/                # ROI · Priority · Risk · Fusion
│   ├── motion/                  # FSM · APF · Speed · Reacquisition
│   ├── cv_fallback/             # Traditional CV backup
│   ├── quant/                   # QAT (α-QAT)
│   ├── compress/                # Channel pruning
│   ├── distill/                 # Knowledge distillation (student builder TBD)
│   └── export/                  # ONNX/TFLite/STM32Cube.AI (TBD)
├── configs/                     # YAML configs (inheritance-based)
├── tools/                       # CLI tools (train/eval/diagnose/cloud)
├── scripts/                     # Platform scripts (.sh + .ps1)
├── tests/                       # Unit tests
├── deploy/openmv/               # OpenMV deployment (MicroPython)
└── requirements.txt
```

---

## Dataset

7 obstacle classes (person, bicycle, car, backpack, suitcase, chair, bottle), ~13,576 training images, YOLO-txt annotation format.

| Split | Count | Source |
|-------|-------|--------|
| Train | 13,576 | COCO 2017 (class-balanced) + KITTI |
| Val | 3,256 | COCO 2017 val |
| Test | — | KITTI (independent generalization, TBD) |

Dataset stats: 73% small targets (<32×32px), 10:1 class imbalance (person 107K vs backpack 10K).

```bash
python tools/prepare_lead_dataset.py    # Generate dataset
python tools/dataset_stats_txt.py       # Dataset diagnostics
```

---

## Ablation Experiments

4 variants for RQ1 (LCA accuracy) + RQ2 (LCA overhead):

| Variant | YAML | LCA | Reduction | Purpose |
|---------|------|-----|-----------|---------|
| `baseline` | `yolo11n_lead.yaml` | ✗ | — | RQ1 control |
| `lca_r16` | `yolo11n_lca_neck_r16.yaml` | ✓ Neck P3 | 16 | RQ1 main |
| `lca_r8` | `yolo11n_lca_neck_r8.yaml` | ✓ Neck P3 | 8 | RQ2 ablation |
| `lca_r32` | `yolo11n_lca_neck_r32.yaml` | ✓ Neck P3 | 32 | RQ2 ablation |

Hard-scene subsets for LCA gain validation:

- `small_targets`: 1,348 images (41.4%) — bbox <32px
- `occlusion`: 464 images (14.25%) — multi-object with IoU>0.3
- `red_background`: 149 images (4.58%) — red channel dominant

---

## Three-Platform Compatibility

| Platform | Role | Script | Notes |
|----------|------|--------|-------|
| Windows | Dev / smoke test | `.ps1` | torchenv, RTX 5060 8GB |
| Linux | Cloud training | `.sh` | AutoDL, RTX 5090 32GB |
| OpenMV H7 Plus | Deployment | MicroPython | Isolated from PyTorch code, TFLite Micro |

---

## Tests

```bash
python tests/test_imports.py         # Import smoke
python tests/test_lca.py             # LCA attention module
python tests/test_data_pipeline.py   # Data pipeline
python tests/test_kalman_filter.py   # Kalman filter
python tests/test_tracker.py         # Multi-object tracking
python tests/test_decision.py        # Decision layer
python tests/test_cv_fallback.py     # CV fallback
```

---

## Who Should Use This?

- Students/teams doing smart car competitions needing visual obstacle avoidance
- Developers researching embedded deep learning deployment
- Researchers needing a lightweight detection + tracking + avoidance pipeline

## Not Suitable For

- High-accuracy general detection (use YOLOv8/RT-DETR directly)
- Multi-camera or LiDAR fusion
- Detecting >7 object classes
- >30 FPS real-time on high-dynamic scenes (MCU is the bottleneck)

---

## Citation

```bibtex
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title  = {Ultralytics YOLO11},
  version = {11.0.0},
  year   = {2024},
  url    = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}

@misc{lead-net,
  title  = {LEAD-Net: Lightweight Edge-aware Attention Detection Network
             for Embedded Obstacle Perception},
  year   = {2026},
  note   = {In preparation}
}
```

## License

MIT (LEAD-Net project) · AGPL-3.0 (Ultralytics YOLO11, requires Enterprise license for closed-source commercial use)
