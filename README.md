# LEAD-Net

**L**ightweight **E**dge-aware **A**ttention **D**etection **Net**work（轻量边缘感知注意力检测网络）

**中文** | [English](README_EN.md)

> 面向嵌入式设备的实时视觉感知系统 —— 让智能小车用单颗 RGB 摄像头就能**看得见、追得上、绕得开**。

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.11-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/cuda-12.8-green)](https://developer.nvidia.com/cuda-toolkit)
[![Ultralytics](https://img.shields.io/badge/ultralytics-8.4-orange)](https://ultralytics.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 项目简介

LEAD-Net 是一套面向资源受限嵌入式平台的轻量级视觉感知系统。它用单颗 RGB 摄像头完成**目标检测 → 追踪 → 避障决策 → 运动规划**，通过极简 UART 协议输出给 STM32 执行 PID 控制。

**部署平台：** OpenMV H7 Plus（STM32H743, 480 MHz Cortex-M7, 1 MB SRAM, 无 NPU）
**训练平台：** 云端 GPU（RTX 5090 32GB）/ 本地开发（RTX 5060 8GB）
**适用场景：** 智能小车竞赛、自动驾驶教学、嵌入式 AI 研究

---

## 系统架构

LEAD-Net 采用**教师–学生蒸馏**架构，弥合高精度训练与 MCU 可部署推理之间的鸿沟：

```text
YOLO11n + LCA(Neck)          教师模型（证明 LCA 有效，RQ1/RQ2）
       │
       ▼  知识蒸馏
   ┌───┴───┐
  FOMO    ST-YOLOXn          学生候选（MCU 可部署）
 (快速)   (精确)
       │
       ▼  INT8 QAT → TFLite → OpenMV
 OpenMV H7 Plus              边缘部署（RQ3/RQ5）
```

### 检测层（训练端）

| 组件 | 说明 |
|------|------|
| 骨干 | YOLO11n（Ultralytics，COCO 预训练 `yolo11n.pt`），2.6M 参数，6.5 GFLOPs |
| LCA | 轻量坐标感知注意力，注入 Neck P3，仅增 ~1.7K 参数 |
| 损失 | Varifocal + DFL + CIoU（Ultralytics 原生） |
| 分配器 | TaskAlignedAssigner（Ultralytics 原生） |

### 决策与控制层（部署端，class-agnostic）

| 组件 | 说明 |
|------|------|
| 决策引擎 | 五层过滤：置信度 → ROI → 距离估计 → DL/CV 融合 → 风险评估 |
| 行为状态机 | SEARCHING → TRACKING → OBSTACLE_DETECTED → AVOIDING → TARGET_REACQUIRE |
| NSA-KF 追踪 | 噪声自适应 Kalman + DIOU 匹配，遮挡 3-5 帧不丢失 |
| APF 避障 | 人工势场法虚拟斥力，输出坐标偏置让 STM32 自然绕行 |
| 速度控制 | 五级 BBox 面积 → 速度映射，靠近目标自动减速 |
| 重拾搜索 | 螺旋搜索 + ROI 扩展 + 路径记忆，丢失后自主找回 |
| CV 兜底 | 传统视觉备份：地面分割 + Blob 检测，DL 失效时接管 |

### UART 输出协议

```text
检测到目标:  "o{cx_offset},d{detected},a{area}\r\n"
目标丢失:    "o0,d0,a0\r\n"
```

| 字段 | 范围 | 含义 |
|------|------|------|
| `o` | -160 ~ +160 | 目标中心相对画面中心的横向偏移（负=左，正=右） |
| `d` | 0 / 1 | 是否检测到目标 |
| `a` | 像素² | 目标面积（bbox 宽×高 或 heatmap 连通域大小） |

---

## 性能指标

| 指标 | 值 | 备注 |
|------|----|------|
| 教师参数量 | 2.6M | YOLO11n + LCA（~1.7K 额外） |
| 教师计算量 | 6.5 GFLOPs | @640 输入（训练）；@416 微调 |
| 输入分辨率 | 416×416 | 云端训练；320×320 本地调试 |
| 检测类别 | 7 类 | person, bicycle, car, backpack, suitcase, chair, bottle |
| 追踪目标数 | ≤ 3 | 可配置 |
| 遮挡容忍 | 3-5 帧 | NSA-KF 预测桥接 |
| mAP@0.5 | *待云端训练* | RTX 5090 训练完成后更新 |
| OpenMV FPS | *待部署* | INT8 TFLite 部署后更新 |

> 旧 SSD-Lite+MobileNetV3 架构 180 epoch 训练后 mAP@0.5 = 8.5%（架构天花板）。YOLO11n 基线在 7 类子集上微调预期可达 70%+。

---

## 快速开始

### 环境要求

- Python 3.11+ · PyTorch 2.11+ · CUDA 12.0+
- 训练：≥8 GB 显存（本地）/ ≥32 GB（云端）
- 推理：CPU 或 OpenMV H7 Plus

```bash
pip install -r requirements.txt
pip install ultralytics  # YOLO11 集成
```

### 本地冒烟测试（5 分钟）

在 100 张样本图上验证完整管线（模型加载 + LCA 注入 + 权重迁移 + 训练 + 评估）：

```bash
# Windows (torchenv)
$env:PYTHONPATH = "."
python tools/smoke_verify_models.py

# Linux
PYTHONPATH=. python tools/smoke_verify_models.py
```

预期输出：`baseline: PASS` + `lca_r16: PASS`

### 云端完整训练（RTX 5090 32GB）

```bash
# 在 AutoDL 云端实例
cd /root/autodl-tmp/LEAD-Net
pip install ultralytics pycocotools
bash scripts/train_cloud.sh smoke     # 1 epoch 验证
bash scripts/train_cloud.sh           # 4 变体 × 180 epoch
```

### 评估

```bash
# 验证集 COCO mAP
PYTHONPATH=. python -c "
from ultralytics import YOLO
m = YOLO('outputs/cloud/runs/lca_r16/weights/best.pt')
metrics = m.val(data='configs/lead_subset_ultralytics.yaml', imgsz=416)
print(f'mAP@0.5: {metrics.box.map50:.4f}')
"

# 构建困难场景子集（遮挡 / 小目标 / 复杂红色背景）
python tools/build_hard_subsets.py
```

---

## 项目结构

```text
LEAD-Net/
├── lead_net/
│   ├── models/
│   │   ├── attention/           # LCA 模块（ultralytics 适配版 + 旧 SSD 兼容）
│   │   ├── yolo/                # YOLO11 集成
│   │   │   ├── lca_adapter.py       # 注册 LCA 到 ultralytics
│   │   │   ├── weight_remapper.py   # LCA 插入后权重键名重映射
│   │   │   ├── data_adapter.py      # lead_subset → ultralytics data.yaml
│   │   │   └── yamls/               # 4 个 YAML（baseline + LCA r=8/16/32）
│   │   ├── backbone.py          # MobileNetV3（旧 SSD 路径，保留消融）
│   │   ├── detection_head.py    # SSD-Lite（旧，保留消融）
│   │   ├── loss.py              # MultiBoxLoss（旧）
│   │   └── lead_net.py          # 模型组装（路由 YOLO / SSD）
│   ├── data/                    # 数据集 · Transforms · DataLoader
│   ├── engine/                  # Trainer · Evaluator · Scheduler
│   ├── tracking/                # NSA-KF · DIOU · MOSSE
│   ├── decision/                # ROI · Priority · Risk · Fusion
│   ├── motion/                  # FSM · APF · Speed · Reacquisition
│   ├── cv_fallback/             # 传统 CV 兜底
│   ├── quant/                   # QAT 量化感知训练
│   ├── compress/                # 通道剪枝
│   ├── distill/                 # 知识蒸馏（学生构建器待做）
│   └── export/                  # ONNX/TFLite/STM32Cube.AI（待做）
├── configs/                     # YAML 配置（继承式）
├── tools/                       # 命令行工具（train/eval/诊断/云端）
├── scripts/                     # 平台脚本（.sh + .ps1）
├── tests/                       # 单元测试
├── deploy/openmv/               # OpenMV 部署（MicroPython）
└── requirements.txt
```

---

## 数据集

7 类障碍物（person, bicycle, car, backpack, suitcase, chair, bottle），约 13,576 张训练图，YOLO-txt 标注格式。

| 划分 | 数量 | 来源 |
|------|------|------|
| Train | 13,576 | COCO 2017（类别均衡采样）+ KITTI |
| Val | 3,256 | COCO 2017 val |
| Test | — | KITTI（独立泛化评估，待做） |

数据集特征：73% 小目标（<32×32px），10:1 类别不平衡（person 107K vs backpack 10K）。

```bash
python tools/prepare_lead_dataset.py    # 生成数据集
python tools/dataset_stats_txt.py       # 数据集诊断
```

---

## 消融实验

4 个变体，对应 RQ1（LCA 精度）+ RQ2（LCA 开销）：

| 变体 | YAML | LCA | Reduction | 用途 |
|------|------|-----|-----------|------|
| `baseline` | `yolo11n_lead.yaml` | ✗ | — | RQ1 对照基准 |
| `lca_r16` | `yolo11n_lca_neck_r16.yaml` | ✓ Neck P3 | 16 | RQ1 主实验 |
| `lca_r8` | `yolo11n_lca_neck_r8.yaml` | ✓ Neck P3 | 8 | RQ2 消融 |
| `lca_r32` | `yolo11n_lca_neck_r32.yaml` | ✓ Neck P3 | 32 | RQ2 消融 |

困难场景子集（LCA 增益验证用）：

- `small_targets`：1,348 张（41.4%）— bbox <32px
- `occlusion`：464 张（14.25%）— 多目标 IoU>0.3
- `red_background`：149 张（4.58%）— 红色通道主导

---

## 三平台兼容

| 平台 | 角色 | 脚本 | 说明 |
|------|------|------|------|
| Windows | 开发 / 冒烟测试 | `.ps1` | torchenv, RTX 5060 8GB |
| Linux | 云端训练 | `.sh` | AutoDL, RTX 5090 32GB |
| OpenMV H7 Plus | 部署 | MicroPython | 与 PyTorch 代码物理隔离，TFLite Micro |

---

## 测试

```bash
python tests/test_imports.py         # 导入冒烟
python tests/test_lca.py             # LCA 注意力模块
python tests/test_data_pipeline.py   # 数据管线
python tests/test_kalman_filter.py   # Kalman 滤波器
python tests/test_tracker.py         # 多目标追踪
python tests/test_decision.py        # 决策层
python tests/test_cv_fallback.py     # CV 兜底
```

---

## 谁应该用这个项目？

- 做智能小车竞赛、需要视觉避障和追踪的学生/团队
- 研究嵌入式深度学习部署的开发者
- 需要轻量级目标检测 + 追踪 + 避障完整 pipeline 的研究者

## 不适合什么场景？

- 高精度通用目标检测（直接用 YOLOv8/RT-DETR）
- 多摄像头、激光雷达融合
- 需要检测超过 7 类物体的场景
- >30 FPS 的高动态实时场景（MCU 是算力瓶颈）

---

## 引用

```bibtex
@software{yolo11_ultralytics,
  author  = {Glenn Jocher and Jing Qiu},
  title   = {Ultralytics YOLO11},
  version = {11.0.0},
  year    = {2024},
  url     = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}

@misc{lead-net,
  title = {LEAD-Net: Lightweight Edge-aware Attention Detection Network
           for Embedded Obstacle Perception},
  year  = {2026},
  note  = {In preparation}
}
```

## 许可证

MIT（LEAD-Net 项目） · AGPL-3.0（Ultralytics YOLO11，闭源商用需购买 Enterprise license）
