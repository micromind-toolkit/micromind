import os
import matplotlib.pyplot as plt
from orion.client import get_experiment

def plot_table(exp_name, base_path):
    experiment = get_experiment(name=exp_name)
    df = experiment.to_pandas(True)
    print(df)
    df.drop(df.iloc[:, 0:6], inplace=True, axis=1)
    df.to_excel(os.path.join(base_path,exp_name) + ".xlsx", index=False)


def plot(x, y, path, plot_type):
    path = os.path.join(path, plot_type + ".jpg")
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
