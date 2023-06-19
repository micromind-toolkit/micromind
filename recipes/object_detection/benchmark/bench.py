from benchmarks import benchmark

# Benchmark on GPU
benchmark(
    model="./weights/microyolo-500-epochs-SGD-2/best.pt",
    imgsz=320,
    half=True,
    device="cpu",
)  # microyolo with phinet
# benchmark(model='yolov8n.pt', imgsz=320, half=False, device='cpu') # yolov8 nano
# benchmark(model="yolov8s.pt", imgsz=320, half=True, device="cpu")  # yolov8 small
