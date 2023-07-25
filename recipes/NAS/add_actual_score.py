import pandas as pd
import os
from utils import plot_table

exp_name = "cifar100_naswot_random_0.05M"
base_path = "/home/majam001/mj/micromind/recipes/NAS/orion_exp/cifar100/random/cifar100_naswot_random_0.05M/"
plot_table(exp_name, base_path)

# Read the existing Excel file
file = os.path.join(base_path, exp_name) + ".xlsx"
existing_data = pd.read_excel(file)

# Initialize a list to store the extracted scores
scores = []

# Read the log file and extract alternate occurrences of Score
with open(base_path + "/info.log", 'r') as log_file:
    lines = log_file.readlines()

score_count = 1
for line in lines:
    if line.startswith('Score:'):
        score_count += 1
        if score_count % 2 == 0:
            score = float(line.split(':')[1].strip())
            scores.append(score)

# Add the scores as a new column to the existing data
existing_data['Actual Score'] = scores

# Save the updated data to the existing Excel file
existing_data.to_excel(file, index=False)