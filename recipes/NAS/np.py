# import numpy as np
# score = 1387.310728
# params = 597770
# target = 50000
# power = np.floor(np.log10(target)) - 1
# #diff = np.abs(params - target)
# diff = 100
# target_obj = (diff / 10 ** power)
# obj = score * 1e-3 - target_obj

# print(power)
# print(diff)
# print(target_obj)
# print(score * 1e-3)
# print(-obj)

# import numpy as np
# import matplotlib.pyplot as plt

# target = 5e6
# score_range = np.arange(500, 1001)
# diff_range = np.arange(1, 100001)

# # Create a meshgrid
# score_mesh, diff_mesh = np.meshgrid(score_range, diff_range)

# # Calculate obj values for each diff value
# power = np.floor(np.log10(target)) - 1
# target_obj = diff_mesh / 10 ** power
# obj_values = -(score_mesh * 1e-3 - target_obj)

# # Plotting the graph
# plt.figure(figsize=(10, 6))
# for i, score in enumerate(score_range):
#     plt.plot(diff_range, obj_values[:, i], label=f'Score: {score}')

# plt.xlabel('diff')
# plt.ylabel('obj')
# #plt.legend()
# plt.title('obj vs. diff')
# plt.grid(True)
# plt.savefig("/home/majam001/mj/micromind/recipes/NAS/orion_exp/fig.jpg")
# plt.cla()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the ranges for each hyperparameter
alpha_range = np.logspace(np.log10(1), np.log10(5), num=100)
beta_range = np.logspace(np.log10(0.5), np.log10(1.3), num=100)
B0_range = np.arange(5, 9)
t_zero_range = np.arange(2, 7)

# Create meshgrid for each pair of hyperparameters
alpha_mesh, beta_mesh = np.meshgrid(alpha_range, beta_range)
alpha_mesh, B0_mesh = np.meshgrid(alpha_range, B0_range)
alpha_mesh, t_zero_mesh = np.meshgrid(alpha_range, t_zero_range)
beta_mesh, B0_mesh = np.meshgrid(beta_range, B0_range)
beta_mesh, t_zero_mesh = np.meshgrid(beta_range, t_zero_range)
B0_mesh, t_zero_mesh = np.meshgrid(B0_range, t_zero_range)

# Define the fixed parameters
nas_args = {
    "target": 1e6,
}


# Calculate the objective function values
def objective_function(score, params):
    power = np.floor(np.log10(nas_args["target"]))
    obj = score * 1e-3 - (np.abs(params - nas_args["target"]) / 10**power)
    return obj


# Evaluate the objective function for each combination of hyperparameters
score = np.linspace(500, 1000, num=100)
params = np.linspace(50000, 10000000, num=100)

objective_values_alpha_beta = objective_function(
    score[:, np.newaxis], params[np.newaxis, :]
)
objective_values_alpha_B0 = objective_function(
    score[:, np.newaxis], params[np.newaxis, :]
)
objective_values_alpha_t_zero = objective_function(
    score[:, np.newaxis], params[np.newaxis, :]
)
objective_values_beta_B0 = objective_function(
    score[:, np.newaxis], params[np.newaxis, :]
)
objective_values_beta_t_zero = objective_function(
    score[:, np.newaxis], params[np.newaxis, :]
)
objective_values_B0_t_zero = objective_function(
    score[:, np.newaxis], params[np.newaxis, :]
)

# Create surface plots
fig = plt.figure(figsize=(10, 8))

# Plot alpha vs. beta
ax1 = fig.add_subplot(231, projection="3d")
ax1.plot_surface(alpha_mesh, beta_mesh, objective_values_alpha_beta)
ax1.set_xlabel("alpha")
ax1.set_ylabel("beta")
ax1.set_zlabel("Objective")

# Plot alpha vs. B0
ax2 = fig.add_subplot(232, projection="3d")
ax2.plot_surface(alpha_mesh, B0_mesh, objective_values_alpha_B0)
ax2.set_xlabel("alpha")
ax2.set_ylabel("B0")
ax2.set_zlabel("Objective")

# Plot alpha vs. t_zero
ax3 = fig.add_subplot(233, projection="3d")
ax3.plot_surface(alpha_mesh, t_zero_mesh, objective_values_alpha_t_zero)
ax3.set_xlabel("alpha")
ax3.set_ylabel("t_zero")
ax3.set_zlabel("Objective")

# Plot beta vs. B0
ax4 = fig.add_subplot(234, projection="3d")
ax4.plot_surface(beta_mesh, B0_mesh, objective_values_beta_B0)
ax4.set_xlabel("beta")
ax4.set_ylabel("B0")
ax4.set_zlabel("Objective")

# Plot beta vs. t_zero
ax5 = fig.add_subplot(235, projection="3d")
ax5.plot_surface(beta_mesh, t_zero_mesh, objective_values_beta_t_zero)
ax5.set_xlabel("beta")
ax5.set_ylabel("t_zero")
ax5.set_zlabel("Objective")

# Plot B0 vs. t_zero
ax6 = fig.add_subplot(236, projection="3d")
ax6.plot_surface(B0_mesh, t_zero_mesh, objective_values_B0_t_zero)
ax6.set_xlabel("B0")
ax6.set_ylabel("t_zero")
ax6.set_zlabel("Objective")

plt.tight_layout()
plt.savefig("/home/majam001/mj/micromind/recipes/NAS/orion_exp/surfaceplot.jpg")
plt.cla()
