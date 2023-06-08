import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Model': ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x'],
    'Size (pixels)': [640, 640, 640, 640, 640],
    'mAPval 50-95': [37.3, 44.9, 50.2, 52.9, 53.9],
    'Speed CPU ONNX (ms)': [80.4, 128.4, 234.7, 375.2, 479.1],
    'Speed A100 TensorRT (ms)': [0.99, 1.20, 1.83, 2.39, 3.53],
    'Params (M)': [3.2, 11.2, 25.9, 43.7, 68.2],
    'FLOPs (B)': [8.7, 28.6, 78.9, 165.2, 257.8]
}

df = pd.DataFrame(data)

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

# Add grid
axs[0].grid(True)
axs[1].grid(True)

# Plot Params (M) vs mAPval 50-95
axs[0].scatter(df['Params (M)'], df['mAPval 50-95'], color='blue')
axs[0].plot(df['Params (M)'], df['mAPval 50-95'], color='blue', linestyle='dashed')

# Annotate points with the model name
for i, model in enumerate(df['Model']):
    axs[0].annotate(model, (df['Params (M)'][i], df['mAPval 50-95'][i]))

axs[0].set_title('Params (M) vs mAPval 50-95')
axs[0].set_xlabel('Params (M)')
axs[0].set_ylabel('mAPval 50-95')

# Plot Speed CPU ONNX (ms) vs mAPval 50-95
axs[1].scatter(df['Speed CPU ONNX (ms)'], df['mAPval 50-95'], color='orange')
axs[1].plot(df['Speed CPU ONNX (ms)'], df['mAPval 50-95'], color='orange', linestyle='dashed')

# Annotate points with the model name
for i, model in enumerate(df['Model']):
    axs[1].annotate(model, (df['Speed CPU ONNX (ms)'][i], df['mAPval 50-95'][i]))

axs[1].set_title('Latency CPU ONNX (ms) vs mAPval 50-95')
axs[1].set_xlabel('Latency CPU ONNX (ms)')
axs[1].set_ylabel('mAPval 50-95')

plt.tight_layout()
plt.savefig('yolov8.png')
plt.show()
