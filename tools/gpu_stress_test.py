"""gpu_stress_test.py — 诊断训练时 GPU 是否真在用 + 找出瓶颈。

跑 50 个 batch 真实训练，每 10 batch 报告：
  - GPU 利用率/显存/功率
  - 数据加载时间 vs GPU 计算时间
  - batch 实际耗时
"""
from __future__ import annotations

import subprocess
import time

import torch
from ultralytics import YOLO


def gpu_stats() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.sm",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        parts = [float(x) for x in out.split(",")]
        return dict(zip(["gpu_util", "mem_util", "mem_used_mib", "mem_total_mib", "power_w", "sm_mhz"], parts))
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 70)
    print("GPU Stress Test: 50 batches @imgsz=320, batch=256")
    print("=" * 70)

    # 构建模型
    model = YOLO("lead_net/models/yolo/yamls/yolo11n_lead.yaml").load("yolo11n.pt")
    model.model.cuda()

    # 检查模型在 GPU 上
    dev = next(model.model.parameters()).device
    print(f"model device: {dev}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")

    # 预热
    print("\n[warmup] 5 batches...")
    dummy = torch.zeros(256, 3, 320, 320, device="cuda")
    for _ in range(5):
        with torch.no_grad():
            _ = model.model(dummy)
    torch.cuda.synchronize()

    print("\n[baseline GPU stats]")
    print(gpu_stats())

    # 真实训练 50 batch（用 ultralytics train 但只跑很少 epoch 不现实，改为手动前向+反向）
    print("\n[stress] 50 forward+backward batches, batch=256...")
    optimizer = torch.optim.SGD(model.model.parameters(), lr=0.01, momentum=0.9)
    model.model.train()

    t0 = time.time()
    for i in range(50):
        batch_t0 = time.time()
        x = torch.randn(256, 3, 320, 320, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.model(x)
            # 提取一个 tensor 做 loss（只测 GPU 压榨，不测正确性）
            if isinstance(out, dict):
                # 取第一个 tensor 值
                loss = next(v for v in out.values() if torch.is_tensor(v)).sum() * 1e-6
            elif isinstance(out, (list, tuple)):
                pred = out[0]
                if isinstance(pred, dict):
                    loss = next(v for v in pred.values() if torch.is_tensor(v)).sum() * 1e-6
                elif torch.is_tensor(pred):
                    loss = pred.sum() * 1e-6
                else:
                    loss = pred[0].sum() * 1e-6
            else:
                loss = out.sum() * 1e-6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        batch_dt = time.time() - batch_t0
        if (i + 1) % 10 == 0:
            stats = gpu_stats()
            print(f"  batch {i+1}/50: {batch_dt*1000:.1f}ms/batch | "
                  f"gpu_util={stats.get('gpu_util')}% mem_used={stats.get('mem_used_mib')}MB "
                  f"power={stats.get('power_w')}W sm={stats.get('sm_mhz')}MHz")

    total = time.time() - t0
    print(f"\n[done] 50 batches in {total:.1f}s = {50/total:.1f} batch/s")
    print(f"final gpu stats: {gpu_stats()}")


if __name__ == "__main__":
    main()
