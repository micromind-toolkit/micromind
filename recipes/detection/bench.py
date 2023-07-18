import os
import shutil

from micromind import benchmark

weight = "_new_start_3_heads_c2f_067_100_epochs"

# Benchmark on GPU
benchmark(
    model="./benchmark/weights/" + weight + "/weights/best.pt",
    imgsz=320,
    device="cpu",
    int8=True,
)  # microyolo with phinet

# rename benchmark.log to weight.log
os.rename("benchmarks.log", weight + ".log")

# move to benchmark folder
shutil.move(weight + ".log", "./benchmark/plots/data/optimized/" + weight + ".log")

# benchmark(model='yolov8n.pt', imgsz=320, half=False, device='cpu') # yolov8 nano
# benchmark(model="yolov8s.pt", imgsz=320, half=True, device="cpu")  # yolov8 small
