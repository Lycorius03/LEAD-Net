#!/bin/bash
# train_cloud.sh — 云端 Linux (AutoDL RTX 5090 32GB) 训练启动脚本
#
# 用法：
#   bash scripts/train_cloud.sh              # 完整 4 变体 180 epoch
#   bash scripts/train_cloud.sh smoke        # 1 epoch 冒烟验证
#   bash scripts/train_cloud.sh baseline lca_r16  # 只跑主对比
#
# 前置准备（云端执行一次）：
#   1. clone 项目到 /root/autodl-tmp/LEAD-Net
#   2. pip install ultralytics pycocotools
#   3. 把 data/lead_subset 数据集传上去（或重新跑 prepare_lead_dataset.py）
#   4. bash scripts/train_cloud.sh smoke  # 先冒烟验证
#   5. bash scripts/train_cloud.sh        # 正式训练

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Python 路径（云端默认 python，如需指定 conda 环境改这里）
PYTHON=${PYTHON:-python}

MODE=${1:-full}
shift 2>/dev/null || true

if [ "$MODE" = "smoke" ]; then
    echo "=== SMOKE TEST (1 epoch) ==="
    $PYTHON tools/cloud_train.py --smoke --variants baseline lca_r16
elif [ "$MODE" = "full" ]; then
    if [ $# -gt 0 ]; then
        echo "=== FULL TRAIN (custom variants: $*) ==="
        $PYTHON tools/cloud_train.py --variants "$@"
    else
        echo "=== FULL TRAIN (all 4 variants, 180 epoch) ==="
        $PYTHON tools/cloud_train.py
    fi
else
    echo "Usage: bash scripts/train_cloud.sh [smoke|full] [variant1 variant2 ...]"
    exit 1
fi

echo ""
echo "=== 训练完成 ==="
echo "权重: outputs/cloud/runs/<variant>/weights/best.pt"
echo "报告: outputs/cloud/report/cloud_train_summary.json"
echo "下载权重和报告回本地后，运行:"
echo "  python tools/stage1c_analyze.py  # 本地分析训练结果"
