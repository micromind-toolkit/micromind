import os
import pandas as pd
import matplotlib.pyplot as plt
import orion.core.io.experiment_builder as experiment_builder
from orion_to_pandas_patch import orion_patch
import torch
import logging

def plot_table(exp_name, base_path):
    experiment = experiment_builder.load(name=exp_name)
    experiment = orion_patch(experiment)
    df = experiment.to_pandas(True)
    df.drop(df.iloc[:, 0:6], inplace=True, axis=1)
    df.to_excel(os.path.join(base_path, exp_name) + ".xlsx", index=False)


def excel_to_df(exp_name, base_path, column):
    df = pd.read_excel(os.path.join(base_path, exp_name) + ".xlsx")
    if column=="all":
        return df
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

def plot_pareto(exp_name, path, plot_type, target):
    # Initialize lists to store the Pareto front and dominated points
    params = excel_to_df(exp_name, path, "params")
    scores = excel_to_df(exp_name, path, plot_type)
    data = excel_to_df(exp_name, path, "all" )
    pareto_front = []
    dominated_points = []
    optimal_points = []

    # Iterate through the data to identify the Pareto front and dominated points
    for i in range(len(scores)):
        if params[i] <= target:
            is_dominated = False
            for j in range(len(scores)):
                if i != j and scores[j] >= scores[i] and params[j] < params[i]:
                    is_dominated = True
                    break
            if is_dominated:
                dominated_points.append((params[i], scores[i]))
            else:
                pareto_front.append((params[i], scores[i]))
                pareto_front.append((params[i], scores[i]))
                optimal_points.append(data.loc[i])

    # Print the list of optimal points with respective parameters
    for i, point in enumerate(optimal_points):
        logging.info(f"Optimal Point {i+1}:")
        logging.info(point)

    # Convert the pareto_front and dominated_points lists to separate lists of params and scores
    pareto_params, pareto_scores = zip(*pareto_front)
    dominated_params, dominated_scores = zip(*dominated_points)

    # Plot the Pareto front and dominated points
    plt.scatter(dominated_params, dominated_scores, color='red', label='Dominated Points')
    plt.scatter(pareto_params, pareto_scores, color='blue', label='Pareto Front')

    # Connect the points on the Pareto front with the attainment surface
    pareto_front_sorted = sorted(pareto_front)
    for i in range(len(pareto_front_sorted) - 1):
        plt.plot([pareto_front_sorted[i][0], pareto_front_sorted[i + 1][0]],
                [pareto_front_sorted[i][1], pareto_front_sorted[i + 1][1]], 'k--')

    # Add the optimal point numbers as a legend
    for i, point in enumerate(optimal_points):
        plt.text(point['params'], point['score'], f"{i+1}")

    # Set labels and title
    plt.xlabel('Params')
    plt.ylabel(plot_type)
    plt.title('Pareto Front Plot')

    # Show the plot
    plt.legend()
    path = os.path.join(path, "pareto_" + plot_type + ".jpg")
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


def add_noise(inputs,noise_factor=0.1):
     noisy = inputs+torch.randn_like(inputs) * noise_factor
     noisy = torch.clip(noisy,0.,1.)
     return noisy


if __name__ == "__main__":
    exp_name = "obj1_seed_42_tpe_0.5M"
    base_path = os.path.join("./orion_exp/cifar10/tpe/", exp_name)
    top_3 = topk_net(base_path, exp_name)
    print(top_3)
