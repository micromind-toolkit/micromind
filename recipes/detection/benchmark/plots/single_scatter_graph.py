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
    # print("ok", column_names, type(column_names))
    column_names.insert(0, "Model")
    column_names.insert(0, "format")
    # add the string "format" at the beginning of the list
    # print(column_names)

    parsed_data = {column_names[i]: [] for i in range(len(column_names))}

    for line in data:
        for i, x in enumerate(line):
            parsed_data[column_names[i]].append(x)

    # Create a dictionary to store the parsed data
    # print(parsed_data)
    return parsed_data


def prepare_plot():
    # Create a figure with only one plot

    _, axs = plt.subplots(1, 1, figsize=(15, 7))
    # Add grid
    axs.grid(True)
    return axs


def plot_data(axs, columns, model):
    pred = pd.DataFrame(columns)
    star_size_ratio = pred["Size (MB)"].max()
    pred["Size (MB)"] = pred["Size (MB)"].apply(lambda x: x * 1000 / star_size_ratio)

    # Plot Inference time (ms/im) vs metrics/mAP50-95(B)
    axs.scatter(
        pred["Inference time (ms/im)"],
        pred["metrics/mAP50-95(B)"],
        color="orange",
        marker=".",
        s=pred["Size (MB)"],
    )
    # Annotate points with the model name
    for i, model in enumerate(pred["Model"]):
        axs.annotate(
            model, (pred["Inference time (ms/im)"][i], pred["metrics/mAP50-95(B)"][i])
        )

    axs.set_title("Latency CPU ONNX (ms) vs metrics/mAP50-95(B)")
    axs.set_xlabel("Latency CPU ONNX (ms)")
    axs.set_ylabel("metrics/mAP50-95(B)")

    return axs


def show_graph(name):
    plt.tight_layout()
    # create timestamp
    import datetime

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # save the plot
    plt.savefig(name + "-benchmark-" + timestamp + ".png")
    plt.show()


if __name__ == "__main__":

    axs = prepare_plot()
    f = ["optimized"]

    for model in f:
        path = "./data/" + model + "/"
        single_bench = read_data(path)
        plot_data(axs, single_bench, model)

    show_graph(f[0])
