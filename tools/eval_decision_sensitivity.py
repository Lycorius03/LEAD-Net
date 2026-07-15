"""避障决策参数敏感性分析工具。

用途（对应 RQ6 / DATA_COLLECTION.md 第七类）：
    - Sweep ROI 比例参数，记录不同配置下的避障成功率
    - Sweep 面积阈值（calibration table），找最优参数组合
    - 记录误触发次数

使用前提：
    - M5 完成后：有训练好的模型可用
    - M7 阶段：有固定测试路线和障碍物摆放方案
    - 本脚本提供参数扫描框架，实际数据采集需配合实拍/仿真

用法：
    python tools/eval_decision_sensitivity.py \
        --config configs/baseline_ssd.yaml \
        --weights outputs/checkpoints/baseline_no_lca.pth \
        --test-scenario scenario.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lead_net.utils import load_config, ExperimentManager
from lead_net.decision import DecisionEngine, DecisionParams, ROIParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEAD-Net 决策参数敏感性分析")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--weights", type=str, default=None,
                   help="模型权重（留空则仅测试决策逻辑，用模拟检测数据）")
    p.add_argument("--test-scenario", type=str, default=None,
                   help="测试场景 JSON（ground truth 障碍物位置+类别）")
    p.add_argument("--output", type=str,
                   default="outputs/experiments/decision_sensitivity.csv")
    return p.parse_args()


def _build_param_grid() -> list[dict[str, Any]]:
    """生成参数扫描网格。

    扫描维度：
        - ROI 水平范围（中央比例）
        - ROI 垂直范围（下半比例）
        - 面积阈值缩放因子（相对默认标定表）

    返回配置列表，每个 dict 含 sweep 参数。
    """
    grid = []
    for h_ratio in [0.5, 0.55, 0.60, 0.65, 0.70]:  # 中央比例
        for v_ratio in [0.40, 0.45, 0.50, 0.55, 0.60]:  # 下半比例
            for area_scale in [0.5, 0.75, 1.0, 1.25, 1.5]:  # 阈值缩放
                h_margin = (1.0 - h_ratio) / 2
                grid.append({
                    "h_start": h_margin,
                    "h_end": 1.0 - h_margin,
                    "v_start": 1.0 - v_ratio,
                    "v_end": 1.0,
                    "area_scale": area_scale,
                })
    return grid


def evaluate_config(
    engine: DecisionEngine,
    detections: list[dict],
    ground_truth: list[dict],
) -> dict[str, float]:
    """评估单组参数：计算避障成功率和误触发率。

    Args:
        engine: 已配置的 DecisionEngine
        detections: 模拟/实际检测结果
        ground_truth: 真实障碍物标注（哪些应该触发避障）

    Returns:
        {"success_rate":, "false_positive_rate":, "false_negative_rate":}
    """
    total = len(ground_truth)
    hits = 0
    fp = 0

    result = engine.decide(detections)

    if result.has_target:
        best_iou = 0.0
        for gt in ground_truth:
            iou = _box_iou(
                [result.x, result.y, result.x + result.a**0.5, result.y + result.a**0.5],
                gt["bbox"],
            )
            best_iou = max(best_iou, iou)
        if best_iou >= 0.3:
            hits += 1
        else:
            fp += 1

    fn = max(0, total - hits)
    return {
        "success_rate": hits / max(total, 1),
        "false_positive_rate": fp / max(total, 1),
        "false_negative_rate": fn / max(total, 1),
    }


def _box_iou(a: list[float], b: list[float]) -> float:
    """xyxy 格式 IoU。"""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-8)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    param_grid = _build_param_grid()
    print(f"[info] testing {len(param_grid)} parameter combinations")

    # 模拟检测数据（实拍后替换为真实检测结果）
    # TODO M7: 使用实际模型对测试视频运行检测
    mock_detections = [
        {"bbox": [160, 220, 80, 80], "score": 0.9, "category_id": 0},
    ]
    mock_gt = [
        {"bbox": [120, 180, 240, 300]},  # xyxy 格式真实障碍物
    ]

    session = ExperimentManager.latest_session("outputs/experiments") or "exp_001"
    csv_path = Path("outputs/experiments") / session / "cross" / "decision_sensitivity.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("h_center_ratio,v_bottom_ratio,area_scale,success_rate,fp_rate,fn_rate\n")
        for params in param_grid:
            engine = DecisionEngine(
                DecisionParams(
                    roi=ROIParams(
                        h_start=params["h_start"], h_end=params["h_end"],
                        v_start=params["v_start"], v_end=params["v_end"],
                    ),
                ),
                tracker=None,
            )
            metrics = evaluate_config(engine, mock_detections, mock_gt)
            f.write(
                f"{1-2*params['h_start']:.2f},{params['v_end']-params['v_start']:.2f},"
                f"{params['area_scale']:.2f},"
                f"{metrics['success_rate']:.3f},{metrics['false_positive_rate']:.3f},"
                f"{metrics['false_negative_rate']:.3f}\n"
            )

    print(f"[info] results → {csv_path}")
    print("[warn] above results use MOCK data — replace with real detections for M7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
