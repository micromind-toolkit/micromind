import pandas as pd
import matplotlib.pyplot as plt

# Load data from Excel file
base_path = "/home/majam001/mj/micromind/recipes/NAS/orion_exp/cifar100/tpe/cifar100_beta_0.01_naswot_tpe_0.05M/"
data = pd.read_excel(base_path + "cifar100_beta_0.01_naswot_tpe_0.05M.xlsx")

# Extract the score and params values
scores = data['Actual Score']
params = data['params']

# Initialize lists to store the Pareto front and dominated points
pareto_front = []
dominated_points = []
optimal_points = []

# Iterate through the data to identify the Pareto front and dominated points
for i in range(len(scores)):
    if params[i] <= 0.07:
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
    print(f"Optimal Point {i+1}:")
    print(point)
    print()

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
    plt.text(point['params'], point['Actual Score'], f"{i+1}")

# Set labels and title
plt.xlabel('Params')
plt.ylabel('Score')
plt.title('Pareto Front Plot')

# Show the plot
plt.legend()
plt.savefig(base_path + "pareto_norm.jpg")
plt.cla()
# Display the plot
plt.show()


