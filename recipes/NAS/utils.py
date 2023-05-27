import os
import pandas as pd
import matplotlib.pyplot as plt
import orion.core.io.experiment_builder as experiment_builder
from orion_to_pandas_patch import orion_patch


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
