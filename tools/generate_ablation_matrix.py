"""消融实验矩阵生成器 — 扫描最新 session 的 baseline/lca 自动汇总。

用法:
    python tools/generate_ablation_matrix.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lead_net.utils import ExperimentManager


def _find_metrics(session_dir: Path, variant: str) -> Path | None:
    train_dir = session_dir / variant / "train"
    if not train_dir.exists():
        return None
    runs = sorted(train_dir.glob("run_*"))
    if not runs:
        return None
    csv_path = runs[-1] / "train_metrics.csv"
    return csv_path if csv_path.exists() else None


def _find_profile(session_dir: Path, variant: str) -> Path | None:
    train_dir = session_dir / variant / "train"
    if not train_dir.exists():
        return None
    runs = sorted(train_dir.glob("run_*"))
    if not runs:
        return None
    csv_path = runs[-1] / "model_profile.csv"
    return csv_path if csv_path.exists() else None


def main() -> int:
    session = ExperimentManager.latest_session("outputs/experiments")
    if not session:
        print("[warn] no experiment session found")
        return 0
    session_dir = Path("outputs/experiments") / session

    configs = [
        {"variant": "baseline", "label": "A", "lca": "N", "kalman": "N", "int8": "N"},
        {"variant": "lca",      "label": "B", "lca": "Y", "kalman": "N", "int8": "N"},
    ]

    rows = []
    for c in configs:
        m_csv = _find_metrics(session_dir, c["variant"])
        p_csv = _find_profile(session_dir, c["variant"])

        best_mAP50, best_mAP, params = "—", "—", "—"
        if m_csv:
            with open(m_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    m50 = float(row.get("val/mAP@0.5", -1) or -1)
                    mcm = float(row.get("val/mAP@0.5:0.95", -1) or -1)
                    if m50 > float(best_mAP50.replace("—", "-1")):
                        best_mAP50 = f"{m50:.4f}"
                    if mcm > float(best_mAP.replace("—", "-1")):
                        best_mAP = f"{mcm:.4f}"
        if p_csv:
            with open(p_csv, newline="", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("total,"):
                        params = f"{int(line.split(',')[1])/1e6:.2f}M"

        rows.append({**c, "mAP50": best_mAP50, "mAP": best_mAP, "params": params})

    # Output
    print(f"=== Ablation Matrix ({session}) ===")
    print(f"{'Config':<8} {'LCA':<5} {'Kalman':<8} {'INT8':<6} {'mAP@0.5':<10} {'mAP@.5:.95':<12} {'Params':<10}")
    print("-" * 60)
    for r in rows:
        print(f"{r['label']:<8} {r['lca']:<5} {r['kalman']:<8} {r['int8']:<6} {r['mAP50']:<10} {r['mAP']:<12} {r['params']:<10}")

    csv_path = session_dir / "cross" / "ablation_matrix.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("config,label,variant,lca,kalman,int8,mAP50,mAP_coco,params\n")
        for r in rows:
            f.write(f"{r['label']},{r['label']},{r['variant']},{r['lca']},{r['kalman']},{r['int8']},{r['mAP50']},{r['mAP']},{r['params']}\n")
    print(f"\n[info] → {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
