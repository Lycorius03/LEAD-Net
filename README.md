# LEAD-Net

**L**ightweight **E**dge-aware **A**ttention **D**etection **Net**work

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.11-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/cuda-12.8-green)](https://developer.nvidia.com/cuda-toolkit)
[![Ultralytics](https://img.shields.io/badge/ultralytics-8.4-orange)](https://ultralytics.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **中文** | [English](README_EN.md)
>
> 面向嵌入式设备的实时视觉感知系统 —— 让智能小车用单颗 RGB 摄像头就能**看得见、追得上、绕得开**。

LEAD-Net 是一套面向资源受限嵌入式平台（OpenMV H7 Plus, STM32H743 @480MHz, 1MB SRAM, 无 NPU）的轻量级视觉感知系统。它用单颗摄像头完成**目标检测 → 追踪 → 避障决策 → 运动规划**，通过极简 UART 协议输出给 STM32 执行 PID 控制。

采用**教师–学生蒸馏**架构：YOLO11n+LCA 教师（证 LCA 有效）→ MCU 友好学生（FOMO/ST-YOLOXn）→ INT8 TFLite → OpenMV 部署。

完整文档请见 [README_zh.md](README_zh.md)（中文）或 [README_EN.md](README_EN.md)（English）。

**快速开始：**

```bash
pip install -r requirements.txt && pip install ultralytics
PYTHONPATH=. python tools/smoke_verify_models.py   # 5 分钟冒烟验证
bash scripts/train_cloud.sh                          # 云端完整训练
```

**关键文档：**

- [docs/HANDOFF_GUIDE.md](docs/HANDOFF_GUIDE.md) — 项目交接指南（从这里开始）
- [docs/CLOUD_TRAIN_GUIDE.md](docs/CLOUD_TRAIN_GUIDE.md) — 云端训练指南
- [docxs/DESIGN_YOLO11_LCA.md](docxs/DESIGN_YOLO11_LCA.md) — 架构设计文档
