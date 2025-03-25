import os
import glob
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =============================================================================
# 1. Helper functions for plotting and analysis
# =============================================================================

def load_attention_csv(csv_path):
    """Load the attention analysis CSV file into a pandas DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def plot_attention_heatmap(df, phrase_type="harmful", out_dir="plots", prompt_key="attack"):
    """
    For a given phrase type (e.g. 'harmful' or 'actionable'), plot a heatmap 
    of average fraction_attention per (layer, head) across all samples.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Group by layer and head and compute the mean fraction_attention
    summary = df[df["phrase_type"] == phrase_type].groupby(["layer", "head"])["fraction_attention"].mean().reset_index()
    
    # Pivot for heatmap: rows=layer, columns=head
    pivot = summary.pivot(index="layer", columns="head", values="fraction_attention")
    
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot, aspect='auto', interpolation="nearest")
    plt.colorbar(label="Avg. Fraction Attention")
    plt.title(f"Heatmap of {phrase_type} Attention (Prompt: {prompt_key})")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.xticks(ticks=range(pivot.shape[1]), labels=pivot.columns)
    plt.yticks(ticks=range(pivot.shape[0]), labels=pivot.index)
    heatmap_path = os.path.join(out_dir, f"attention_heatmap_{prompt_key}_{phrase_type}.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heatmap to: {heatmap_path}")

def plot_attention_by_layer(df, phrase_type="harmful", out_dir="plots", prompt_key="attack"):
    """
    Create bar plots for each layer showing the distribution of fraction_attention per head.
    """
    os.makedirs(out_dir, exist_ok=True)
    layers = sorted(df["layer"].unique())
    for layer in layers:
        sub_df = df[(df["layer"] == layer) & (df["phrase_type"] == phrase_type)]
        summary = sub_df.groupby("head")["fraction_attention"].agg(["mean", "std", "median"]).reset_index()
        plt.figure(figsize=(8, 4))
        plt.bar(summary["head"], summary["mean"], yerr=summary["std"], capsize=5)
        plt.xlabel("Head")
        plt.ylabel("Mean Fraction Attention")
        plt.title(f"Layer {layer} ({phrase_type}) - Prompt: {prompt_key}")
        plt.xticks(summary["head"])
        plot_path = os.path.join(out_dir, f"attention_layer_{layer}_{prompt_key}_{phrase_type}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved bar plot for layer {layer} to: {plot_path}")

def load_activation_analysis_results(pt_path):
    """Load the minimal activation analysis results from a .pt file."""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"File not found: {pt_path}")
    results = torch.load(pt_path, map_location="cpu")
    return results

def plot_pca_scatter(pca_results, out_dir="plots", prompt_type="attack"):
    """
    For each layer and aggregation method, plot the PCA(2D) scatter plots.
    pca_results should be a dict with structure: 
      { layer: { agg_method: np.array(shape=(N,2)) } }
    """
    os.makedirs(out_dir, exist_ok=True)
    for layer, methods in pca_results.items():
        for method, coords in methods.items():
            if coords.shape[0] == 0:
                continue
            plt.figure(figsize=(6, 5))
            plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=20)
            plt.title(f"PCA Scatter - Layer {layer}, Method: {method}\nPrompt: {prompt_type}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            scatter_path = os.path.join(out_dir, f"pca_layer_{layer}_{method}_{prompt_type}.png")
            plt.savefig(scatter_path)
            plt.close()
            print(f"Saved PCA scatter plot to: {scatter_path}")

def run_kmeans_on_pca(coords, n_clusters=2):
    """
    Optionally run k-means clustering on PCA coordinates and return the cluster labels.
    """
    if coords.shape[0] < n_clusters:
        return np.zeros(coords.shape[0], dtype=int)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(coords)
    return labels

def plot_kmeans_clusters(coords, labels, layer, method, prompt_type, out_dir="plots"):
    """Plot the PCA scatter with k-means cluster labels overlaid."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", alpha=0.7, s=20)
    plt.title(f"KMeans Clusters - Layer {layer}, Method: {method}\nPrompt: {prompt_type}")
    plt.xlabel("PC1")
