# train_cloud.ps1 — Windows 本地调试用（与 train_cloud.sh 逻辑一致）
#
# 用法：
#   .\scripts\train_cloud.ps1                      # 完整 4 变体 180 epoch
#   .\scripts\train_cloud.ps1 -Mode smoke          # 1 epoch 冒烟
#   .\scripts\train_cloud.ps1 -Variants baseline lca_r16  # 只跑主对比
#
# 注意：本地 RTX 5060 8GB 跑完整训练会显存不足，仅用 smoke 模式验证
# 完整训练请在云端 Linux 用 train_cloud.sh

param(
    [string]$Mode = "full",
    [string[]]$Variants = @()
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = $PWD.Path

$Python = "F:\.anaconda\envs\torchenv\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

if ($Mode -eq "smoke") {
    Write-Host "=== SMOKE TEST (1 epoch) ==="
    if ($Variants.Count -gt 0) {
        & $Python tools/cloud_train.py --smoke --variants $Variants
    } else {
        & $Python tools/cloud_train.py --smoke --variants baseline lca_r16
    }
} elseif ($Mode -eq "full") {
    if ($Variants.Count -gt 0) {
        Write-Host "=== FULL TRAIN (custom variants: $Variants) ==="
        & $Python tools/cloud_train.py --variants $Variants
    } else {
        Write-Host "=== FULL TRAIN (all 4 variants, 180 epoch) ==="
        & $Python tools/cloud_train.py
    }
} else {
    Write-Host "Usage: .\scripts\train_cloud.ps1 [-Mode smoke|full] [-Variants baseline lca_r16 ...]"
    exit 1
}

Write-Host ""
Write-Host "=== 训练完成 ==="
Write-Host "权重: outputs/cloud/runs/<variant>/weights/best.pt"
Write-Host "报告: outputs/cloud/report/cloud_train_summary.json"
