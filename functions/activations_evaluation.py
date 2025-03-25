# file: functions/analysis_evaluation.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_analysis_results(path):
    """
    Load the analysis results from a .pt file.
    Returns a dictionary with keys like "aggregator_outputs" and "pca_2d".
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Analysis results file not found at: {path}")
    return torch.load(path, map_location="cpu")

def plot_pca_scatter(pca_results, title="PCA Scatter Plot"):
    """
    Plots a scatter plot of PCA(2D) results.
    
    pca_results: dict, with keys being layer numbers (or strings) and values being a dict mapping
                 aggregation methods (e.g. 'last_token', 'last_few', 'mean_all') to np.array of shape (N,2).
    """
    num_layers = len(pca_results)
    fig, axes = plt.subplots(num_layers, 1, figsize=(8, num_layers * 4), squeeze=False)
    for i, (layer, methods) in enumerate(pca_results.items()):
        ax = axes[i][0]
        for method, data in methods.items():
            if data.shape[0] == 0:
                continue
            ax.scatter(data[:, 0], data[:, 1], label=method, alpha=0.6)
        ax.set_title(f"Layer {layer}")
        ax.legend()
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def perform_kmeans_on_activations(activations, n_clusters=3):
    """
    Performs KMeans clustering on the given activation data.
    
    activations: numpy array of shape (N, hidden_dim)
    n_clusters: int, number of clusters to form.
    
    Returns:
      - labels: cluster assignments (numpy array)
      - silhouette: silhouette score (float) for the clustering.
    """
    if activations.shape[0] < n_clusters:
        return None, None
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(activations)
    silhouette = silhouette_score(activations, labels)
    return labels, silhouette

def evaluate_clustering(aggregator_outputs, n_clusters=3):
    """
    Evaluates clustering on the aggregated activations using KMeans for each layer and aggregation method.
    
    aggregator_outputs: dict mapping layer -> method -> numpy array of shape (N, hidden_dim)
    
    Returns:
      A dict of the form { layer: { method: silhouette_score } }
    """
    scores = {}
    for layer, methods in aggregator_outputs.items():
        scores[layer] = {}
        for method, data in methods.items():
            if data.size == 0 or data.shape[0] < n_clusters:
                scores[layer][method] = None
            else:
                _, score = perform_kmeans_on_activations(data, n_clusters=n_clusters)
                scores[layer][method] = score
    return scores

def compare_prompt_types(result_paths):
    """
    Given a dictionary mapping prompt type names to analysis results file paths,
    load each and compute overall activation means per layer and aggregation method.
    
    Returns a dict:
      { prompt_type: { layer: { method: mean_activation_vector (np.array) } } }
    """
    summary = {}
    for prompt_type, path in result_paths.items():
        res = load_analysis_results(path)
        means = {}
        aggregator_outputs = res.get("aggregator_outputs", {})
        for layer, methods in aggregator_outputs.items():
            means[layer] = {}
            for method, data in methods.items():
                if data.size == 0:
                    means[layer][method] = None
                else:
                    means[layer][method] = np.mean(data, axis=0)
        summary[prompt_type] = means
    return summary

def run_activation_evaluation(analysis_result_path, n_clusters=3):
    """
    Runs a complete evaluation on a single analysis result file:
      - Loads the analysis results.
      - Plots the PCA(2D) scatter plots.
      - Performs KMeans clustering on the "mean_all" aggregated activations for each layer.
    
    Parameters:
      analysis_result_path: str, path to your analysis_results.pt file.
      n_clusters: int, number of clusters for KMeans.
      
    Returns:
      A dictionary mapping each layer to its clustering silhouette score (using the "mean_all" method).
    """
    results = load_analysis_results(analysis_result_path)
    aggregator_outputs = results.get("aggregator_outputs", {})
    pca_results = results.get("pca_2d", {})
    prompt_type = results.get("prompt_type", "Unknown")
    
    # Plot PCA results.
    plot_pca_scatter(pca_results, title=f"PCA Scatter Plot for prompt type '{prompt_type}'")
    
    # Evaluate clustering using the "mean_all" aggregation.
    clustering_scores = {}
    for layer, methods in aggregator_outputs.items():
        data = methods.get("mean_all", None)
        if data is not None and data.shape[0] >= n_clusters:
            _, score = perform_kmeans_on_activations(data, n_clusters=n_clusters)
            clustering_scores[layer] = score
        else:
            clustering_scores[layer] = None
    
    print("Clustering silhouette scores (using 'mean_all' aggregation):")
    for layer, score in clustering_scores.items():
        print(f"  Layer {layer}: {score}")
    
    return clustering_scores
