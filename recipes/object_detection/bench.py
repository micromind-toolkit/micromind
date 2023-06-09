from modules.benchmarks import benchmark

# Benchmark on GPU
#benchmark(model='../runs/detect/train3/weights/best.pt', imgsz=320, half=True, device='cpu') # microyolo with phinet
#benchmark(model='yolov8n.pt', imgsz=320, half=True, device='cpu') # yolov8 with module n
benchmark(model='yolov8s.pt', imgsz=320, half=True, device='cpu') # yolov8 with module s

#gather the data


# save the data in the data.csv file