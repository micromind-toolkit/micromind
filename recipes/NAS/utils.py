import os
import matplotlib.pyplot as plt
from orion.client import get_experiment


def plot_table(exp_name, param, scores, target_objs, base_path):
    experiment = get_experiment(name=exp_name)
    df = experiment.to_pandas(True)
    print(df)
    df.drop(df.iloc[:, 0:6], inplace=True, axis=1)
    df["Parameters"] = param
    df["Score"] = scores
    df["Target Objective"] = target_objs
    df.to_excel(base_path + ".xlsx", index=False)


def plot(x, y, path, plot_type):
    path = os.path.join(path + "_" + plot_type + ".jpg")
    plt.scatter(x, y, color="blue", alpha=0.5)
    if plot_type == "objective":
        plt.ylabel("Objective")
    elif plot_type == "naswot":
        plt.ylabel("NASWOT Score")
    elif plot_type == "loss":
        plt.ylabel("Val loss")
    elif plot_type == "target":
        plt.ylabel("Target Objective")
    else:
        "This type of plot is not defined!"
    plt.savefig(path)
    plt.cla()
