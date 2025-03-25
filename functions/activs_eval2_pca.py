# file: functions/compare_pca.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_analysis_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Analysis results file not found at: {path}")
    return torch.load(path, map_location="cpu")

def overlay_pca_scatter(result_paths, aggregation_method="mean_all"):
    """
    Overlays PCA scatter plots for a specific aggregation method across multiple prompt types.
    s
    result_paths: dict mapping prompt type (string) to analysis result file path.
    aggregation_method: which aggregation method to plot.
    """
    plt.figure(figsize=(10, 8))
    for prompt, path in result_paths.items():
        results = load_analysis_results(path)
        pca_results = results.get("pca_2d", {})
        # For simplicity, overlay results from a selected layer (e.g., layer_0)
        # You can modify this to loop over layers.
        layer_key = "layer_0"
        if layer_key in pca_results and aggregation_method in pca_results[layer_key]:
            data = pca_results[layer_key][aggregation_method]
            if data.shape[0] > 0:
                plt.scatter(data[:, 0], data[:, 1], alpha=0.6, label=prompt)
    plt.title(f"PCA Overlay Scatter Plot for {aggregation_method} (Layer 0)")
    plt.legend()
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.show()
