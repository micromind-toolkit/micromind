import torch
import os
import csv

class ActivationHook:
    def __init__(self):
        # Data structure to store the activations
        self.activations = {}

    def __call__(self, module, input, output, name=None):
        # Handle different output types
        if isinstance(output, torch.Tensor):
            activation_data = output.detach().cpu().numpy()
            self.activations[name] = activation_data
        elif isinstance(output, (list, tuple)):
            self.activations[name] = []
            for idx, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    activation_data = out.detach().cpu().numpy()
                    self.activations[name].append(activation_data)
                else:
                    self.activations[name].append(None)  # Handle non-tensor elements gracefully
        else:
            # If output is neither Tensor nor list/tuple, handle as unsupported
            self.activations[name] = None

    def clear(self):
        """Clears the stored activations."""
        self.activations = {}


class CSV_ActivationHook:
    def __init__(self):

        self._activation_file = "outputs/activations.csv"
        # Ensure the directory existsWarning: 
        os.makedirs(os.path.dirname(self._activation_file), exist_ok=True)

        # Open the file and write headers (we'll append data later)
        self._activation_file_handle = open(self._activation_file, "w", newline="")
        self._csv_writer = csv.writer(self._activation_file_handle)

        self._csv_writer.writerow(["Layer Name", "Layer Type", "Activation Shape", "Activations"])  # Headers


    def __call__(self, module, input, output, name):
        if isinstance(output, torch.Tensor):
            activation_data = output.detach().cpu().numpy()
            self._csv_writer.writerow([name, module.__class__.__name__, activation_data.shape, activation_data.tolist()])
        elif isinstance(output, (list, tuple)):
            for idx, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    activation_data = out.detach().cpu().numpy()
                    self._csv_writer.writerow([f"{name}[{idx}]", module.__class__.__name__, activation_data.shape, activation_data.tolist()])
        # Flush to ensure data is written to the file
        self._activation_file_handle.flush()
    
    def close_file(self):
        self._activation_file_handle.close()
        print(f"Activation file {self._activation_file} closed.")
