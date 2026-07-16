# LEAD-Net 训练系统修复计划 (v3)

> 日期：2026-07-16
> 状态：[OK] 已完成执行
> 版本：v3

---

## 背景

LEAD-Net 首次云端训练（exp_001, LCA variant）发现严重问题：
- 训练4个epoch后中断，checkpoints文件夹完全为空
- loss完全不下降（12.68 → 12.67）
- 学习率极低（2e-4而非目标的1e-3）
- 梯度爆炸（grad_norm ~ 497,000）

## 根因分析（7项Bug）

| # | Bug | 严重度 | 文件 |
|---|-----|--------|------|
| 1 | 无定期checkpoint保存，中断后权重全丢 | FATAL | trainer.py:193 |
| 2 | Stage1->Stage2 Backbone未解冻，全程冻结 | FATAL | trainer.py:101-110 |
| 3 | Cosine调度器T_max=总迭代数但按epoch调用 | CRITICAL | trainer.py:317-318 |
| 4 | Warmup实现错误，warmup_start_factor未使用 | CRITICAL | trainer.py:264-267 |
| 5 | Loss归一化clamp(min=1)导致梯度爆炸 | SEVERE | loss.py:111 |
| 6 | LLRD组initial_lr未保存 | MODERATE | llrd.py:58 |
| 7 | CPU模式下GPU pipeline仍尝试CUDA传输 | MODERATE | trainer.py:257 |

## 修复方案

### 新文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `lead_net/engine/scheduler.py` | ~130 | Warmup+Cosine调度器（SequentialLR模式） |
| `lead_net/engine/checkpoint.py` | ~200 | Checkpoint保存/恢复管理 |
| `tests/test_scheduler.py` | ~120 | 调度器单元测试（4项） |
| `tests/test_checkpoint.py` | ~100 | Checkpoint单元测试（4项） |
| `docxs/RESEARCH.md` | ~150 | 研究资料清单 |
| `docxs/PLAN.md` | 本文件 | 修复计划 |

### 修改文件

| 文件 | 变更概述 |
|------|---------|
| `trainer.py` | 集成checkpoint定期保存、正确Stage过渡、迭代级调度器 |
| `loss.py` | clamp(min=1) → clamp(min=10) + NaN安全检查 |
| `llrd.py` | 添加freeze_backbone/unfreeze_backbone辅助函数 |
| `evaluator.py` | 添加TXT数据集→COCO GT转换支持 |
| `__init__.py` | 导出新模块 |
| `train.py` | 添加--smoke/--resume/--device参数，batch_size自适应 |
| `lead_subset.yaml` | 添加eta_min_factor配置项 |

## 验证结果

### 单元测试
```
[PASS] test_warmup_phase
[PASS] test_cosine_phase
[PASS] test_multi_group_independence
[PASS] test_short_warmup_boundary
[PASS] test_save_load_latest
[PASS] test_save_best
[PASS] test_snapshot_cleanup
[PASS] test_missing_checkpoint
```

### 冒烟测试（2图片，1 epoch，CPU）
- [OK] 训练循环完整运行
- [OK] Loss正常（train=12.81, val=1344.40）
- [OK] Grad norm降到15.17（之前497,000）
- [OK] Warmup正确：LR从1e-6升至1e-3
- [OK] COCO评估正常运行
- [OK] Checkpoint保存成功（best.pt 5.6MB + latest.pt 11.2MB）

### 关键指标对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Grad Norm | ~497,000 | ~15 |
| Head LR (warmup后) | ~2e-4 (错误) | 1e-3 (正确) |
| 调度器步进频率 | 每epoch | 每iteration |
| Checkpoint保存 | 无 (训练末尾一次性) | 每epoch + best + 快照 |
| Stage1→2 Backbone | 未解冻 | 正确解冻并重建优化器 |

## 使用指南

### 冒烟测试
```bash
python tools/train.py --config configs/train_lca.yaml --smoke
```

### 正常训练
```bash
python tools/train.py --config configs/train_lca.yaml
```

### 恢复训练
```bash
python tools/train.py --config configs/train_lca.yaml --resume
```

### 仅CPU调试
```bash
python tools/train.py --config configs/train_lca.yaml --smoke --device cpu
```
