"""推理入口脚本（M1 骨架阶段）。

依据：
    - docs/ARCHITECTURE.md §配置驱动。
    - docs/PLAN.md §M1：本地跑通 Baseline 训练/推理全流程。
    - 本文件 M1 骨架阶段：可 import 不报错，实际推理因占位 head/未解码而 abort。

用法（M1 推理阶段补全后）：
    python tools/infer.py --config configs/baseline_ssd.yaml --image path/to/img.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lead_net.models import build_lead_net
from lead_net.utils import load_config, resolve_paths_in, get_nested


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net 推理入口")
    p.add_argument("--config", required=True, type=str, help="配置文件路径")
    p.add_argument("--image", type=str, default=None, help="单张图片路径（待 M1 后续步骤支持）")
    p.add_argument("--checkpoint", type=str, default=None, help="模型权重路径")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = resolve_paths_in(cfg)

    dev = get_nested(cfg, "device.type", "cuda")
    device = torch.device(dev if (dev == "cuda" and torch.cuda.is_available()) else "cpu")

    model = build_lead_net(cfg)

    # 推断逻辑：M1 骨架阶段不实现真实推理（占位 head + 无 NMS 解码）
    raise NotImplementedError("推理逻辑待 M1 推理阶段实现（M1 第一步不跑推理）。")


if __name__ == "__main__":
    raise SystemExit(main())