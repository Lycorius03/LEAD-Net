# configs/ — 配置驱动说明

所有超参数（学习率、batch size、输入分辨率、类别筛选、LCA 开关等）必须以 YAML 文件形式写入本目录，代码中禁止硬编码。

## 文件

- `default.yaml`         — 公共默认值（路径占位、类别映射表骨架），被其他配置继承/覆盖。
- `baseline_ssd.yaml`   — M1 Baseline 配置：MobileNetV3-Small + SSD-Lite，`use_lca: false`（锁死，对应消融实验 RQ1/RQ2 的对照基线）。

## 设计原则（来自 docs/ARCHITECTURE.md §配置驱动）

- 路径以占位符或相对路径形式写在 yaml，由 `lead_net/utils/config.py` 用 `pathlib.Path` 解析；代码不写死绝对路径。
- LCA 模块必须可通过 `use_lca: true/false` 开关，服务于消融实验 RQ1/RQ2。
- 任何超参调整，必须在 `docs/CHANGELOG.md` 记录理由与依据。

## 读取方式

```python
from lead_net.utils.config import load_config
cfg = load_config("configs/baseline_ssd.yaml")
```

## 版本

本目录配置版本随代码一同演进。每次云端训练产出 checkpoint 时，须与对应 config 快照 yaml 一起打包（见 ARCHITECTURE.md §跨平台适配规范 4）。
