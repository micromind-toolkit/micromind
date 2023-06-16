import numpy as np
import random
import matplotlib.pyplot as plt
from micromind import PhiNet
from logsynflow import compute_logsynflow
from utils import get_mod_data
from orion_search import data_loader, get_params

# Define the search space
search_space = {
    "alpha": np.logspace(np.log10(1), np.log10(5), 100),
    "beta": np.logspace(np.log10(0.5), np.log10(1.3), 100),
    "B0": np.arange(5, 9),
    "t_zero": np.arange(2, 7),
    "res": np.array([64]),
}

# Data
loader_train, loader_eval = data_loader(
    image_size=(3, 64, 64), batch_size=64, dset="cifar10"
)
inputs, targets = get_mod_data(
    loader_train=loader_train, num_classes=10, samples_per_class=6, device=0
)

# Generate param combinations
hparams = []
for alpha in search_space["alpha"]:
    for beta in search_space["beta"]:
        for B0 in search_space["B0"]:
            for t_zero in search_space["t_zero"]:
                hparams.append((alpha, beta, B0, t_zero, 64))
hparams = np.array(hparams)

# Generate 100 random param combinations
random.seed(42)  # Set a seed for reproducibility
indices = np.random.choice(len(hparams), size=100, replace=False)
hparams = hparams[indices]
print(hparams.shape)

# Calculate scores for each param combination
scores = []
params = []
for hparam in hparams:
    alpha, beta, B0, t_zero, res = hparam

    model = PhiNet(
        input_shape=[3, res, res],
        alpha=alpha,
        beta=beta,
        t_zero=t_zero,
        num_layers=B0,
        squeeze_excite=True,
        h_swish=False,
        include_top=True,
        num_classes=10,
    )
    model.cuda(0)
    param = get_params(model, 64)
    params.append(param)

    score = compute_logsynflow(model, inputs, device=0, mode="param")
    scores.append(score)

scores = np.array(scores)
print(scores)
params = np.array(params)

# Plot the distribution of scores vs. params
plt.scatter(params, scores)
plt.xlabel("Scores")
plt.ylabel("Params")
plt.title("Distribution of Score vs. Params")
plt.savefig("/home/majam001/mj/micromind/recipes/NAS/dist.jpg")
plt.cla()
