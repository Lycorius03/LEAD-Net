# LEAD-Net

**Lightweight Edge-aware Attention Detection Network**

面向资源受限边缘设备（OpenMV H7 Plus）的轻量级通用障碍物感知与避障系统。

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.11-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 概述

LEAD-Net 是一个用于智能小车动态避障的轻量化视觉感知系统。核心创新包括：

- **LCA**（Lightweight Coordinate-aware Attention）—— 基于 Coordinate Attention（CVPR 2021）的轻量注意力模块，嵌入 MobileNetV3-Small 骨干，参数开销 < 1%
- **SSD-Lite 检测头** —— 多尺度深度可分离卷积，2475 个锚框，覆盖 16-317 px 目标尺度
- **Class-Agnostic 三层决策** —— 置信度过滤 → ROI 空间约束 → 面积代理紧急度优先级，不依赖类别白名单，对任意前景物体有效
- **DL + 传统 CV 混合感知** —— 深度学习精细检测 + 颜色聚类地面分割兜底，提升未知物体鲁棒性
- **Kalman 多目标追踪** —— 8 维状态向量匀速模型，贪心 IoU 匹配，Track 生命周期管理

### 系统架构

```
Camera → MobileNetV3-Small + LCA + SSD-Lite → Detections
  → [DL Path] ─┐
               ├→ Decision Engine (3-layer) → Priority Target
  → [CV Path] ─┘  Confidence → ROI → Area Proxy
  → Kalman Tracking → Obstacle State → STM32 UART
```

## 快速开始

### 环境

```bash
# Python 3.11, PyTorch 2.11+cu128, CUDA 12.8
pip install -r requirements.txt
```

### 训练

```bash
# Baseline（无 LCA）
python tools/train.py --config configs/baseline_ssd.yaml

# +LCA 消融对照
python tools/train.py --config configs/lca_ssd.yaml

# 快速测试（限制样本数）
python tools/train.py --config configs/baseline_ssd.yaml --max-samples 100 --epochs 3
```

### 评估

```bash
python tools/eval.py --config configs/baseline_ssd.yaml --weights outputs/checkpoints/baseline_no_lca.pth
```

### 锚框分析

```bash
python tools/inspect_anchors.py --config configs/baseline_ssd.yaml
```

## 项目结构

```
LEAD-Net/
├── configs/                # YAML 配置（继承式，配置驱动）
│   ├── default.yaml        #   公共默认配置
│   ├── baseline_ssd.yaml   #   Baseline 消融对照（LCA=false）
│   └── lca_ssd.yaml        #   +LCA 实验组（LCA=true）
├── lead_net/               # 源码包
│   ├── models/             #   Backbone / LCA / SSD-Lite Head / Loss
│   ├── data/               #   COCO Dataset / Transforms / DataLoader
│   ├── engine/             #   Trainer / Evaluator / MetricsCollector
│   ├── tracking/           #   KalmanFilter / MultiTargetTracker
│   ├── decision/           #   DecisionEngine / ROIFilter / Priority / Fusion
│   ├── cv_fallback/        #   GroundSegmenter / BlobDetector / CvFallback
│   └── utils/              #   Config / Path 工具
├── tools/                  # 入口脚本
│   ├── train.py            #   训练入口
│   ├── eval.py             #   独立评估入口
│   ├── inspect_anchors.py  #   锚框分析工具
│   └── download_coco.py    #   COCO 数据集下载
├── tests/                  # 单元测试（7 文件，55+ 项）
├── deploy/                 # OpenMV 部署（M6）
├── configs/                # YAML 配置文件
└── requirements.txt        # Python 依赖
```

## 模块依赖

```
models/     ← 内部闭环（backbone → attention → head）
data/       ← 内部闭环（coco → transforms → dataloader）
engine/     ← TYPE_CHECKING 解耦（零运行时跨模块依赖）
tracking/   ← 零依赖（纯 numpy）
decision/   ← 零依赖（纯 Python）
cv_fallback/← 零依赖（纯 numpy）
utils/      ← 零依赖
```

## 配置

所有超参数通过 `configs/*.yaml` 管理，支持 `inherit` 继承合并。关键配置项：

| 配置段 | 说明 |
|--------|------|
| `model` | Backbone 选型、LCA 开关/缩减比、Detection Head 参数 |
| `data` | 输入分辨率 320、ImageNet 归一化、数据增强策略 |
| `train` | epochs/batch_size/lr/优化器/调度器/梯度裁剪 |
| `eval` | mAP 阈值、NMS 参数、评估间隔 |
| `tracking` | 最大追踪数 N=3、min_hits、T_lost、IoU 阈值 |
| `decision` | 置信度阈值、ROI 比例、优先级权重、融合参数 |
| `cv_fallback` | 地面分割颜色阈值、blob 面积/密度过滤 |

## 测试

```bash
python tests/test_imports.py        # 导入冒烟测试
python tests/test_lca.py            # LCA 模块
python tests/test_kalman_filter.py  # Kalman 滤波器
python tests/test_tracker.py        # 多目标追踪器
python tests/test_decision.py       # 三层决策引擎
python tests/test_cv_fallback.py    # 传统 CV 兜底
python tests/test_data_pipeline.py  # 数据管线
```

## 里程碑

| 阶段 | 状态 | 内容 |
|------|------|------|
| M1 | ✅ | Baseline（MobileNetV3-Small + SSD-Lite）全流程跑通 |
| M2 | ✅ | LCA 注意力模块设计与集成 |
| M3 | ✅ | Detection Head 锚框核查与尺度调整 |
| M4 | ✅ | Kalman 多目标追踪 + 指标采集系统 |
| M5 | ⬜ | 云端全量训练 + INT8 量化 + TFLite 转换 |
| M6 | ⬜ | OpenMV 部署与实时性验证 |
| M7 | ⬜ | 消融实验数据采集与论文图表 |
| M8 | ⬜ | STM32 通信联调 |
| M9 | ✅ | 三层避障决策 + 传统 CV 兜底（代码已完成） |

## 数据集

COCO 2017（全 80 类），train2017 ~80K 张 / val2017 5K 张，自动过滤不完整下载。

## 引用

```bibtex
@misc{lead-net,
  title={LEAD-Net: Lightweight Edge-aware Attention Detection Network
         for Embedded Obstacle Perception},
  year={2026},
  note={In preparation}
}
```

## 许可证

MIT License
