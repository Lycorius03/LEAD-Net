# 训练策略研究资料清单

> 收集时间：2026-07-16
> 目的：为 LEAD-Net (MobileNetV3-Small + LCA + SSD-Lite) 量身定制现代化训练方案
> 状态：[OK] 已分析，待执行

---

## 资料清单

### 1. 现代化目标检测训练范式（2024-2025）

**来源**：Bag of Freebies (arXiv:1902.04103)、YOLOv5/v8 社区实践、PyTorch 官方文档

**核心发现**：
- Warmup + Cosine Annealing 已成为目标检测训练的**事实标准**，相比传统 Step Decay 提升约 +0.5~1.8 mAP
- 学习率调度器必须**按 iteration 步进**（非按 epoch），否则 Cosine 曲线会被压缩
- Warmup 正确公式：`lr = base_lr × (warmup_start_factor + (1-warmup_start_factor) × step/warmup_steps)`
- Warmup 时长应为总训练步数的 5-10%，对 fine-tuning 任务建议 10-15%
- `SequentialLR(LinearLR + CosineAnnealingLR)` 是 PyTorch 官方推荐的最简洁实现方式

**有效性评估**：[OK] 高 — 来自工业界验证的标准实践

---

### 2. MobileNetV3 微调策略

**来源**：MobileNetV3 官方论文、PyTorch 社区实践

**核心发现**：
- Fine-tuning 时 Backbone 学习率应比从头训练低 **10×**
- BatchNorm 层在 fine-tuning 时需特殊处理：保持 `track_running_stats=True`，使用预训练统计量
- 分阶段训练（先冻结 backbone 训练 head，再联合训练）是最佳实践
- LLRD (Layer-wise Learning Rate Decay) 对 MobileNet 系列有效，推荐衰减因子 0.8-0.9

**有效性评估**：[OK] 高 — MobileNetV3 官方推荐

---

### 3. 训练中断恢复与 Checkpoint 策略

**来源**：RF-DETR 框架实践、PyTorch 社区最佳实践

**核心发现**：
- 每个 epoch 结束后必须保存 checkpoint（含 model + optimizer + scheduler + epoch）
- 最佳策略："latest.pt"（每 epoch 更新）+ "best.pt"（mAP 最优时更新）+ 间隔快照（每 N epoch 保留）
- 训练恢复时需加载完整状态（optimizer state_dict, scheduler state_dict）
- 同步写入时注意：先写到临时文件再 rename，防止写入中断导致文件损坏

**有效性评估**：[OK] 高 — 工业标准

---

### 4. 损失函数数值稳定性

**来源**：SSD 论文 (arXiv:1512.02325)、社区实践

**核心发现**：
- MultiBox Loss 中 `num_pos` 可能极小时（如 batch 中无目标），除以 `num_pos` 会导致 loss 爆炸
- 标准做法：`total_pos = num_pos.sum().float().clamp(min=1)` — 当前代码已做，但 `min=1` 仍可能导致单样本 loss 过大
- 建议：增加 `min=10` 或使用 batch-level normalization
- 分类损失中的 Hard Negative Mining 可能导致梯度集中在少数样本上
- Focal Loss (gamma=2.0) 可作为 CrossEntropy 的替代，自动处理类别不平衡

**有效性评估**：[OK] 高 — 社区广泛验证的解决方案

---

### 5. 学习率调度器实现陷阱

**来源**：PyTorch issue #117540、Sebastian Raschka PyCon 2024

**核心发现**：
- `CosineAnnealingLR(T_max=N)` 的 `T_max` 必须与 `scheduler.step()` 调用频率匹配
- 按 epoch 调用时 `T_max` 应为总 epoch 数；按 iteration 调用时 `T_max` 为总迭代数
- **当前 LEAD-Net 的致命错误**：按 epoch 调用 scheduler.step() 但 T_max=总迭代数
- 解决方案：改用按 iteration 调用 + `SequentialLR(LinearLR, CosineAnnealingLR)` 
- 或改用 `T_max=epochs` 按 epoch 调用

**有效性评估**：[OK] 高 — 直接对应当前 bug

---

### 6. 小模型 + 小数据集训练技巧

**来源**：社区实践、YOLO 训练文档

**核心发现**：
- Label Smoothing (eps=0.1) 对小模型泛化有帮助，提升约 +0.3~0.6 mAP
- Mosaic 增强对小目标检测有效，但最后 10% epoch 应关闭（避免分布偏移）
- EMA decay 建议 0.9998-0.9999（当前设置正确）
- Gradient Clipping 在 warmup 结束后可适当放宽（如 20.0 而非 10.0）
- 对于 batch_size=64，建议使用 SyncBN 或确保 BN momentum 足够大

**有效性评估**：[OK] 中高 — 部分已验证，部分为经验性建议

---

### 7. 优化器选择对比

**来源**：社区实践、论文对比

| 优化器 | 适用场景 | 推荐学习率 | 备注 |
|--------|---------|-----------|------|
| SGD + Momentum (0.9) | 检测任务传统选择 | 1e-3 ~ 1e-2 | 需要精心调参 |
| AdamW | 收敛更快，小模型友好 | 1e-4 ~ 1e-3 | weight_decay 解耦 |
| SGD + Nesterov | 检测任务最佳实践 | 1e-3 | 当前选择，保留 |

**有效性评估**：[OK] 当前 SGD+Nesterov 选择合理，不需要更换

---

## 综合结论

### 必须修复的 Bug（7项）
1. **定期保存 Checkpoint** — 每 epoch + 最佳 mAP + 周期性快照
2. **Stage 1 结束后解冻 Backbone** — 重新调用 `_setup_optimizer(freeze_backbone=False)`
3. **Cosine 调度器按 iteration 步进** — 使用 `SequentialLR(LinearLR, CosineAnnealingLR)`
4. **Warmup 正确实现** — 从 `base_lr × warmup_start_factor` 开始
5. **Loss 归一化改进** — `clamp(min=10)` 防止梯度爆炸
6. **学习率配置验证** — 确保 LLRD 五组 LR 正确传递
7. **验证集评估修复** — 确保 mAP 计算在 eval_interval 触发

### 新增改进建议（3项）
1. 训练恢复功能（从 checkpoint resume）
2. 训练监控增强（grad_norm 记录 5 个参数组的独立值）
3. 冒烟测试模式（1-2 张图片快速验证管线）

---

## 参考文献

- SSD: arXiv:1512.02325
- SSD-Lite: arXiv:1801.04381
- Bag of Freebies: arXiv:1902.04103
- MobileNetV3: arXiv:1905.02244
- Focal Loss: arXiv:1708.02002
- PyTorch LR Scheduler: https://pytorch.org/docs/stable/optim.html
- RF-DETR Checkpoint: https://github.com/roboflow/rf-detr
