# file: functions/activation_variance.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_analysis_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return torch.load(path, map_location="cpu")

def compute_variance_per_layer(analysis_results, aggregation_method="mean_all"):
    """
    Computes the variance (per feature averaged over samples) of the aggregated activations.
    Returns a dictionary mapping layer -> variance value.
    """
    aggregator_outputs = analysis_results.get("aggregator_outputs", {})
    variances = {}
    for layer, methods in aggregator_outputs.items():
        data = methods.get(aggregation_method, None)
        if data is not None and data.size > 0:
            # Compute variance over samples for each feature, then take the mean variance
            variances[layer] = np.mean(np.var(data, axis=0))
        else:
            variances[layer] = None
    return variances

def plot_variance(variances, title="Activation Variance per Layer"):
    layers = sorted(variances.keys(), key=lambda x: int(x.split('_')[1]) if isinstance(x, str) and x.startswith("layer_") else int(x))
    var_values = [variances[layer] for layer in layers]
    plt.figure(figsize=(8, 4))
    plt.plot(layers, var_values, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Mean Variance")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
