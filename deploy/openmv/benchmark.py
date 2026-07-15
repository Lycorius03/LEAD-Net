# deploy/openmv/benchmark.py
# OpenMV H7 Plus 实时性基准测试（MicroPython 脚本）
#
# 用途（对应 RQ5 / DATA_COLLECTION.md 第六类）：
#     - 测量推理延迟（均值+标准差+最大值）
#     - 测量端到端延迟（采集→预处理→推理→后处理→串口发送）
#     - 记录 FPS 和内存占用
#
# 使用方法（在 OpenMV IDE 中打开并运行）：
#     1. 将 LEAD-Net INT8 TFLite 模型放入 SD 卡
#     2. 修改 MODEL_PATH 指向模型文件
#     3. 运行本脚本
#     4. 串口终端查看输出，复制到 CSV
#
# 注意：
#     - 本脚本仅为框架，实际推理需要 TFLite Micro 模型（M5 产出）
#     - 当前可作为"传统CV兜底"模块的性能基准测试
#     - 测量次数建议 >= 100 次（统计意义）

import sensor
import time
import gc
import math
from pyb import UART

# ====== 配置 ======
MODEL_PATH = "lead_net_int8.tflite"  # SD 卡上的模型路径
INPUT_SIZE = 320
N_MEASUREMENTS = 100  # 测量次数
WARMUP_FRAMES = 10    # 预热帧数（不计入统计）

# ====== 初始化 ======
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)  # 320x240
sensor.skip_frames(time=2000)

uart = UART(3, 115200)  # P4 TX

gc.enable()

# ====== 测量数据容器 ======
inference_times = []   # 推理延迟（ms）
end_to_end_times = []  # 端到端延迟（ms）
fps_samples = []

print("=== OpenMV H7 Plus Benchmark ===")
print(f"  Measurements: {N_MEASUREMENTS}")
print(f"  Input size:   {INPUT_SIZE}x{INPUT_SIZE}")
print(f"  Free memory:  {gc.mem_free() / 1024:.0f} KB")

# 预热
print("  Warming up...")
for i in range(WARMUP_FRAMES):
    img = sensor.snapshot()
    # TODO M5: 实际 TFLite 推理
    # net.classify(img) 或 net.detect(img)

print("  Benchmarking...")

for i in range(N_MEASUREMENTS):
    t0 = time.ticks_us()

    # 1. 采集
    img = sensor.snapshot()

    # 2. 预处理（resize + normalize）
    # TODO M5: img.resize(INPUT_SIZE, INPUT_SIZE)

    # 3. 推理
    t1 = time.ticks_us()
    # TODO M5: result = net.forward(img)
    t2 = time.ticks_us()

    # 4. 后处理（模拟 decode + NMS 开销）
    # TODO M5: decoded = decode(result)

    # 5. 串口发送
    # uart.write(...)

    t3 = time.ticks_us()

    # 记录
    inference_us = time.ticks_diff(t2, t1)
    end_to_end_us = time.ticks_diff(t3, t0)
    inference_times.append(inference_us / 1000.0)
    end_to_end_times.append(end_to_end_us / 1000.0)

    # 每 20 帧报告一次
    if (i + 1) % 20 == 0:
        fps = 1000.0 / max(end_to_end_us / 1000.0, 1.0)
        fps_samples.append(fps)
        print(f"  [{i+1}/{N_MEASUREMENTS}] "
              f"inf={inference_us/1000:.1f}ms "
              f"e2e={end_to_end_us/1000:.1f}ms")

# ====== 统计 ======
def stats(arr):
    if not arr:
        return (0, 0, 0)
    mean = sum(arr) / len(arr)
    var = sum((x - mean)**2 for x in arr) / len(arr)
    std = math.sqrt(var)
    return (mean, std, max(arr))

inf_mean, inf_std, inf_max = stats(inference_times)
e2e_mean, e2e_std, e2e_max = stats(end_to_end_times)
avg_fps = 1000.0 / e2e_mean if e2e_mean > 0 else 0
mem_free = gc.mem_free() / 1024.0

print()
print("=== Results ===")
print(f"  Inference (ms):  mean={inf_mean:.1f}  std={inf_std:.1f}  max={inf_max:.1f}")
print(f"  End-to-end (ms): mean={e2e_mean:.1f}  std={e2e_std:.1f}  max={e2e_max:.1f}")
print(f"  FPS:             {avg_fps:.1f}")
print(f"  Free memory:     {mem_free:.0f} KB")
print()

# CSV 格式（复制到 outputs/experiments/openmv_benchmark.csv）
print("=== CSV (copy to PC) ===")
tag = MODEL_PATH.replace(".tflite", "")
print(f"{tag},{inf_mean:.1f},{inf_std:.1f},{inf_max:.1f},{e2e_mean:.1f},{e2e_std:.1f},{e2e_max:.1f},{avg_fps:.1f},{mem_free:.0f}")
