import os
import pandas as pd
import matplotlib.pyplot as plt

# read the data from folder /data


def parse_text_file(file_name):
    data = []
    with open(file_name, "r", encoding="utf-8") as file:
        lines = file.readlines()

        if len(lines) < 4:
            print("The file does not contain enough lines.")
            return

        # Extract the first line (column names) and the line starting with "2"
        column_names = lines[2].strip().split("  ")
        data_line = lines[5].split()[1:]
        data_line = [float(x) if can_be_float(x) else x for x in data_line]
        print([type(x) for x in data_line])

        print(data_line)
        data.append(data_line)

        return column_names, data_line


def can_be_float(x):
    try:
        _ = float(x)
        return True
    except TypeError and ValueError:
        return False


def read_data():
    data = []
    # read files in the dir /data
    for file in os.listdir("data"):
        if file.endswith(".log"):
            column_names, data_line = parse_text_file(os.path.join("data", file))
            model = file.replace("benchmarks_", "").replace(".log", "")
            data_line.insert(1, model)
            data.append(data_line)
    print("ok", column_names, type(column_names))
    column_names.insert(0, "Model")
    column_names.insert(0, "format")
    # add the string "format" at the beginning of the list
    print(column_names)

    parsed_data = {column_names[i]: [] for i in range(len(column_names))}

    for line in data:
        for i, x in enumerate(line):
            parsed_data[column_names[i]].append(x)

    # Create a dictionary to store the parsed data
    print(parsed_data)
    return parsed_data


def plot_data(data):

    star_size=400

    df = pd.DataFrame(data)
    print(df)

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Add grid
    axs[0].grid(True)
    axs[1].grid(True)

    # Plot Size (MB) vs metrics/mAP50-95(B)
    axs[0].scatter(df["Size (MB)"][1:], df["metrics/mAP50-95(B)"][1:], color="blue")
    axs[0].scatter(df["Size (MB)"][:1], df["metrics/mAP50-95(B)"][:1], color="purple", marker="*", s=star_size)
    axs[0].plot(
        df["Size (MB)"][1:], df["metrics/mAP50-95(B)"][1:], color="blue", linestyle="dashed"
    )

    # Annotate points with the model name
    for i, model in enumerate(df["Model"]):
        axs[0].annotate(model, (df["Size (MB)"][i], df["metrics/mAP50-95(B)"][i]))

    axs[0].set_title("Size (MB) vs metrics/mAP50-95(B)")
    axs[0].set_xlabel("Size (MB)")
    axs[0].set_ylabel("metrics/mAP50-95(B)")

    # Plot Inference time (ms/im) vs metrics/mAP50-95(B)
    axs[1].scatter(
        df["Inference time (ms/im)"][1:], df["metrics/mAP50-95(B)"][1:], color="orange"
    )
    axs[1].scatter(
        df["Inference time (ms/im)"][:1], df["metrics/mAP50-95(B)"][:1], color="purple", marker="*", s=star_size
    )
    axs[1].plot(
        df["Inference time (ms/im)"][1:],
        df["metrics/mAP50-95(B)"][1:],
        color="orange",
        linestyle="dashed",
    )

    # Annotate points with the model name
    for i, model in enumerate(df["Model"]):
        axs[1].annotate(
            model, (df["Inference time (ms/im)"][i], df["metrics/mAP50-95(B)"][i])
        )

    axs[1].set_title("Latency CPU ONNX (ms) vs metrics/mAP50-95(B)")
    axs[1].set_xlabel("Latency CPU ONNX (ms)")
    axs[1].set_ylabel("metrics/mAP50-95(B)")

    plt.tight_layout()
    plt.savefig("yolov8.png")
    plt.show()


r = read_data()
plot_data(r)
