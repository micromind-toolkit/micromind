import os
import shutil

from micromind import benchmark


def bench(weight=""):

    # Benchmark on GPU
    benchmark(
        model="./benchmark/weights/_new_start/" + weight + "/weights/best.pt",
        imgsz=320,
        device="cpu",
        half=True,
        int8=True,
    )  # microyolo with phinet

    # rename benchmark.log to weight.log
    os.rename("benchmarks.log", weight + ".log")

    # move to benchmark folder
    shutil.move(weight + ".log", "./benchmark/plots/data/half/" + weight + ".log")

    # benchmark(model='yolov8n.pt', imgsz=320, half=False, device='cpu') # yolov8 nano
    # benchmark(model="yolov8s.pt", imgsz=320, half=True, device="cpu")  # yolov8 small


if __name__ == "__main__":

    weights = os.listdir("./benchmark/weights/_new_start/")[0:1]
    for i in weights:
        print("benching: " + i + "...")
        bench(i)
