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


def read_data(folder):
    data = []
    # read files in the dir /data
    for file in os.listdir(folder):
        if file.endswith(".log"):
            column_names, data_line = parse_text_file(os.path.join(folder, file))
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


def plot_data(pre, trained, micro):

    star_size = 400

    predf = pd.DataFrame(pre)
    print(predf)

    traindf = pd.DataFrame(trained)
    print(traindf)

    microdf = pd.DataFrame(micro)
    print(microdf)

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Add grid
    axs[0].grid(True)
    axs[1].grid(True)

    # Plot Size (MB) vs metrics/mAP50-95(B)
    axs[0].scatter(
        predf["Size (MB)"],
        predf["metrics/mAP50-95(B)"],
        color="blue",
    )
    axs[0].plot(
        predf["Size (MB)"],
        predf["metrics/mAP50-95(B)"],
        color="blue",
        linestyle="dashed",
    )

    axs[0].scatter(
        traindf["Size (MB)"],
        traindf["metrics/mAP50-95(B)"],
        color="blue",
    )
    axs[0].plot(
        traindf["Size (MB)"],
        traindf["metrics/mAP50-95(B)"],
        color="blue",
        linestyle="dashed",
    )

    axs[0].scatter(
        microdf["Size (MB)"],
        microdf["metrics/mAP50-95(B)"],
        color="purple",
        marker="*",
        s=star_size,
    )

    # Annotate points with the model name
    for i, model in enumerate(predf["Model"]):
        axs[0].annotate(model, (predf["Size (MB)"][i], predf["metrics/mAP50-95(B)"][i]))

    for i, model in enumerate(traindf["Model"]):
        axs[0].annotate(
            model, (traindf["Size (MB)"][i], traindf["metrics/mAP50-95(B)"][i])
        )

    for i, model in enumerate(microdf["Model"]):
        axs[0].annotate(
            model, (microdf["Size (MB)"][i], microdf["metrics/mAP50-95(B)"][i])
        )

    axs[0].set_title("Size (MB) vs metrics/mAP50-95(B)")
    axs[0].set_xlabel("Size (MB)")
    axs[0].set_ylabel("metrics/mAP50-95(B)")

    # Plot Inference time (ms/im) vs metrics/mAP50-95(B)
    axs[1].scatter(
        predf["Inference time (ms/im)"],
        predf["metrics/mAP50-95(B)"],
        color="orange",
    )
    axs[1].plot(
        predf["Inference time (ms/im)"],
        predf["metrics/mAP50-95(B)"],
        color="orange",
        linestyle="dashed",
    )

    # Plot Inference time (ms/im) vs metrics/mAP50-95(B)
    axs[1].scatter(
        traindf["Inference time (ms/im)"],
        traindf["metrics/mAP50-95(B)"],
        color="orange",
    )
    axs[1].plot(
        traindf["Inference time (ms/im)"],
        traindf["metrics/mAP50-95(B)"],
        color="orange",
        linestyle="dashed",
    )

    axs[1].scatter(
        microdf["Inference time (ms/im)"],
        microdf["metrics/mAP50-95(B)"],
        color="purple",
        marker="*",
        s=star_size,
    )

    # Annotate points with the model name
    for i, model in enumerate(predf["Model"]):
        axs[1].annotate(
            model, (predf["Inference time (ms/im)"][i], predf["metrics/mAP50-95(B)"][i])
        )

    for i, model in enumerate(traindf["Model"]):
        axs[1].annotate(
            model,
            (traindf["Inference time (ms/im)"][i], traindf["metrics/mAP50-95(B)"][i]),
        )

    for i, model in enumerate(microdf["Model"]):
        axs[1].annotate(
            model,
            (microdf["Inference time (ms/im)"][i], microdf["metrics/mAP50-95(B)"][i]),
        )

    axs[1].set_title("Latency CPU ONNX (ms) vs metrics/mAP50-95(B)")
    axs[1].set_xlabel("Latency CPU ONNX (ms)")
    axs[1].set_ylabel("metrics/mAP50-95(B)")

    plt.tight_layout()
    plt.savefig("yolov8.png")
    plt.show()


if __name__ == "__main__":
    pre = read_data("./data/pre-trained/")
    trained = read_data("./data/trained/")
    yolo_micro = read_data("./data/compare/")

    plot_data(pre, trained, yolo_micro)
