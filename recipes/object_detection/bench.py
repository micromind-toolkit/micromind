from modules.benchmarks import benchmark

# Benchmark on GPU
benchmark(model='./runs/microyolo-500-epochs/best.pt', imgsz=320, half=True, device='cpu') # microyolo with phinet
# benchmark(model='yolov8n.pt', imgsz=320, half=False, device='cpu') # yolov8 with module n
# benchmark(model="yolov8s.pt", imgsz=320, half=True, device="cpu")  # yolov8 with module s