# LEAD-Net

Lightweight Edge-aware Attention Detection Network

面向资源受限边缘设备（OpenMV H7 Plus）的 **目标追踪 + 障碍规避 + 重拾机制** 轻量级视觉感知系统。

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.11-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 概述

LEAD-Net 是一个用于智能小车 **目标追踪与动态避障** 的轻量化视觉感知系统。
核心创新：**轻量级视觉目标追踪与重拾机制（Lightweight Target Tracking and Re-acquisition Mechanism）**

四大核心能力：

1. **稳定追踪目标** — NSA-KF 自适应卡尔曼 + DIOU 匹配
2. **检测前方障碍** — SSD-Lite + Class-Agnostic 感知
3. **绕过障碍** — APF 人工势场法自主避障
4. **重新找到并继续追踪** — 螺旋搜索 + ROI 扩展 + 路径记忆（**[CORE]** 论文创新点）

核心技术栈：

- **LCA**（Lightweight Coordinate-aware Attention）—— 面向避障特化的轻量坐标注意力模块，集成无参特征梯度边缘引导 (Edge Guidance)、自适应通道压缩 (Adaptive Reduction Ratio)、中下方固定空间先验 (Obstacle Prior Mask) 与残差门控 (Residual Attention Gate)，相比经典 CA 仅增加 1 个标量参数。
- **SSD-Lite 检测头** —— 多尺度深度可分离卷积，2475 个锚框
- **NSA-KF + DIOU 追踪** —— 噪声自适应 Kalman + 距离感知 IoU 匹配，遮挡后重捕获率提升
- **五状态 FSM** —— SEARCHING → TRACKING → OBSTACLE_DETECTED → AVOIDING → TARGET_REACQUIRE
- **视觉伺服** —— 五级 BBox Area → Speed 精细映射
- **MOSSE 辅助追踪** — 相关滤波轻量追踪，DL 检测失败时桥接 3-5 帧

### 系统架构（v4）

```text
Camera → MobileNetV3-Small + LCA + SSD-Lite → Detections
  → [DL Path: 目标检测 + 障碍检测]
  → [FSM State Machine: 五状态追踪-避障-重拾]
     SEARCHING → TRACKING → OBSTACLE_DETECTED → AVOIDING → TARGET_REACQUIRE
  → NSA-KF Tracking (DIOU 匹配 + 自适应噪声)
  → Speed Controller (五级面积-速度映射)
  → Reacquisition Engine (螺旋搜索 + ROI 扩展)
  → UART Output (x,y,a 格式不变)
```

**v4 新增模块**：

| 模块 | 文件 | 说明 |
| ---- | ---- | ---- |
| ReacquisitionEngine | `motion/reacquisition.py` | 螺旋搜索 + ROI 扩展重拾 |
| NSA-KF | `tracking/kalman_filter.py` | 噪声自适应 Kalman |
| MOSSETracker | `tracking/mosse_tracker.py` | 轻量相关滤波辅助追踪 |
| Graph Fusion | `graph/fusion.py` | Conv-BN-ReLU 算子融合 |

## 快速开始

### 环境

```bash
# Python 3.11, PyTorch 2.11+cu128, CUDA 12.8
pip install -r requirements.txt
```

### 数据准备

```bash
# COCO 7 类子集 + KITTI 道路场景 → YOLO-txt (~20,000 张)
python tools/prepare_lead_dataset.py
```

### 训练

```bash
# 冒烟测试（2 张图 1 epoch，验证管线）
python tools/train.py --config configs/train_lca.yaml --smoke

# Baseline（无 LCA）
python tools/train.py --config configs/train_baseline.yaml

# +LCA 消融对照
python tools/train.py --config configs/train_lca.yaml

# 从 checkpoint 恢复训练
python tools/train.py --config configs/train_lca.yaml --resume

# CPU 调试模式
python tools/train.py --config configs/train_lca.yaml --smoke --device cpu
```

**训练策略（v3 更新）：**

- **两阶段**：冻结 Backbone → LLRD 联合训练
- **调度器**：Linear Warmup + Cosine Annealing（按 iteration 步进）
- **定期保存**：每 epoch 保存 latest.pt + best.pt + 10 epoch 快照
- **完整恢复**：latest.pt 含 model + optimizer + scheduler 状态
- **数值稳定**：改进的 loss 归一化 + NaN 安全检查
- **显存自适应**：batch_size=auto 自动探测最优值

### 评估

```bash
python tools/eval.py --config configs/train_baseline.yaml --weights outputs/checkpoints/baseline_no_lca.pth
```

### 锚框分析

```bash
python tools/inspect_anchors.py --config configs/lead_subset.yaml
```

## 项目结构

```text
LEAD-Net/
├── configs/                # YAML 配置（继承式，配置驱动）
│   ├── default.yaml        #   公共默认配置
│   ├── lead_subset.yaml    #   COCO 7类子集 + KITTI 道路场景
│   ├── train_baseline.yaml #   Baseline 消融对照（LCA=false）
│   └── train_lca.yaml      #   +LCA 实验组（LCA=true）
├── lead_net/               # 源码包
│   ├── models/             #   Backbone / LCA / SSD-Lite Head / Loss
│   ├── data/               #   COCO / YOLO-txt Dataset / Transforms / DataLoader
│   ├── engine/             #   Trainer / Evaluator / Scheduler / Checkpoint / Metrics
│   ├── tracking/           #   [v4] NSA-KF / MultiTargetTracker (DIOU) / MOSSETracker
│   ├── decision/           #   DecisionEngine / ROIFilter / Priority / Fusion / Risk
│   ├── motion/             #   [v4] 五状态FSM / ReacquisitionEngine / APF / SpeedController
│   ├── graph/              #   [v4] Conv-BN-ReLU 算子融合
│   ├── cv_fallback/        #   GroundSegmenter / BlobDetector / CvFallback
│   └── utils/              #   Config / Path 工具
├── tools/                  # 入口脚本
│   ├── train.py            #   训练入口
│   ├── eval.py             #   独立评估入口
│   ├── inspect_anchors.py  #   锚框分析工具
│   ├── prepare_lead_dataset.py # COCO 7类子集 + KITTI 格式转换
│   └── download_coco.py    #   COCO 数据集下载
├── tests/                  # 单元测试（9 文件，60+ 项）
├── deploy/                 # OpenMV 部署（M6）
└── requirements.txt        # Python 依赖
```

## 模块依赖

```text
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
| -------- | ------ |
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
| ------ | ------ | ------ |
| M1 | [x] | Baseline（MobileNetV3-Small + SSD-Lite）全流程跑通 |
| M2 | [x] | LCA 注意力模块设计与集成 |
| M3 | [x] | Detection Head 锚框核查与尺度调整 |
| M4 | [x] | Kalman 多目标追踪 + 指标采集系统 |
| M5 | [ ] | 云端全量训练 + INT8 量化 + TFLite 转换 |
| M6 | [ ] | OpenMV 部署与实时性验证 |
| M7 | [ ] | 消融实验数据采集与论文图表 |
| M8 | [ ] | STM32 通信联调 |
| M9 | [x] | 三层避障决策 + 传统 CV 兜底 |
| M10 | [x] | 运动规划：APF 避障 + 自适应减速 + 稳定跟随 |
| M11 | [x] | **v4 全栈优化：五状态FSM + NSA-KF + DIOU + 螺旋重拾 + 图融合** |

## 数据集

**COCO 7 类子集 + KITTI 道路场景**（~20,000 张），类别均衡采样，面向 Fine-tuning 优化：

| ID | 类别 | COCO 采样上限 | 说明 |
| -- | ---- | ------------- | ---- |
| 0 | person | 3,500 | KITTI 也映射为 person |
| 1 | bicycle | 2,500 | KITTI Cyclist 映射为 bicycle |
| 2 | car | 3,500 | KITTI Car+Van+Truck 映射为 car |
| 3 | backpack | 2,000 | COCO 独有 |
| 4 | suitcase | 1,500 | COCO 独有 |
| 5 | chair | 2,500 | COCO 独有 |
| 6 | bottle | 2,500 | COCO 独有 |

**数据集划分**：

| Split | 图片数 | 说明 |
| ----- | ------ | ---- |
| Train | ~20,000 | COCO ~17,500（7类均衡）+ KITTI 2,500（道路场景） |
| Val | ~2,200 | COCO val（含目标类别的全部图片） |
| Test | ~4,900 | KITTI 独立泛化测试（不参与训练/调参） |

- 标注格式：YOLO-txt（归一化 cxcywh）
- 数据增强：mosaic(0.5) + random_horizontal_flip + color_jitter  
- Backbone：MobileNetV3-Small（ImageNet 预训练），Detection Head：SSD-Lite（COCO 预训练）
- 设计原则：不训练完整 COCO（11.8 万张），仅 Fine-tune ~2 万张让模型适应任务
- 论文指标：Test 集上报告 mAP、FPS、追踪误差；自己录制 5-10 段视频展示实际部署效果

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
