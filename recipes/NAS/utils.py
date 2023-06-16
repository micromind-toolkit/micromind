import os
import pandas as pd
import matplotlib.pyplot as plt
import orion.core.io.experiment_builder as experiment_builder
from orion_to_pandas_patch import orion_patch
import torch


def plot_table(exp_name, base_path):
    experiment = experiment_builder.load(name=exp_name)
    experiment = orion_patch(experiment)
    df = experiment.to_pandas(True)
    df.drop(df.iloc[:, 0:6], inplace=True, axis=1)
    df.to_excel(os.path.join(base_path, exp_name) + ".xlsx", index=False)


def excel_to_df(exp_name, base_path, column):
    df = pd.read_excel(os.path.join(base_path, exp_name) + ".xlsx")
    return df[column].tolist()


def plot(exp_name, path, plot_type):
    x = excel_to_df(exp_name, path, "params")
    y = excel_to_df(exp_name, path, plot_type)
    plt.scatter(x, y, color="blue", alpha=0.5)

    if plot_type == "objective":
        plt.ylabel("Objective")
    elif plot_type == "score":
        plt.ylabel("NASWOT Score")
    elif plot_type == "loss":
        plt.ylabel("Val loss")
    elif plot_type == "target_obj":
        plt.ylabel("Target Objective")
    else:
        "This type of plot is not defined!"

    path = os.path.join(path, plot_type + ".jpg")
    plt.savefig(path)
    plt.cla()


def topk_net(base_path, exp_name, k):
    data = pd.read_excel(os.path.join(base_path, exp_name) + ".xlsx")
    sorted_data = data.sort_values("objective", ascending=True)
    top_k = sorted_data.head(k)
    return top_k


# Gives a modified train batch with k samples of each class
def get_mod_data(loader_train, num_classes, samples_per_class, device):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(loader_train)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx : idx + 1], targets[idx : idx + 1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    x = torch.cat([torch.cat(_, 0) for _ in datas]).to(device)
    y = torch.cat([torch.cat(_) for _ in labels]).view(-1).to(device)
    return x, y


if __name__ == "__main__":
    exp_name = "obj1_seed_42_tpe_0.5M"
    base_path = os.path.join("./orion_exp/cifar10/tpe/", exp_name)
    top_3 = topk_net(base_path, exp_name)
    print(top_3)
