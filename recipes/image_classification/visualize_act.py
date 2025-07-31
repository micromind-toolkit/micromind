import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast  # To safely evaluate nested lists

# Function to process and plot activation histograms
def plot_activation_histogram(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Iterate through each layer
    for index, row in df.iterrows():
        layer_name = row['Layer Name']
        layer_type = row['Layer Type']
        activation_shape = row['Activation Shape']
        activations = row['Activations']

        # Convert activations string to a nested list of floats
        try:
            activation_values = np.array(ast.literal_eval(activations))  # Safely parse nested lists
            flattened_activations = activation_values.flatten()  # Flatten the array
        except (ValueError, SyntaxError):
            print(f"Error processing activations for layer: {layer_name}")
            continue

        # Compute statistics
        min_value = np.min(flattened_activations)
        max_value = np.max(flattened_activations)
        mean_value = np.mean(flattened_activations)

        # Plot histogram
        plt.figure(figsize=(8, 6))
        plt.hist(flattened_activations, bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.title(f"Histogram of Activations\nLayer: {layer_name} ({layer_type})")
        plt.xlabel("Activation Values")
        plt.ylabel("Frequency")

        # Annotate with statistics and shape
        stats_text = (f"Activation Shape: {activation_shape}\n"
                      f"Min: {min_value:.4f}\nMax: {max_value:.4f}\nMean: {mean_value:.4f}")
        plt.annotate(stats_text, xy=(0.7, 0.6), xycoords='axes fraction', fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='lightyellow'))

        # Show or save the plot
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Input CSV file path
    csv_file_path = 'outputs/activations.csv'  # Replace with your actual file path

    # Run the function
    plot_activation_histogram(csv_file_path)
