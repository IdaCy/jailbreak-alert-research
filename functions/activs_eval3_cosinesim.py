# file: functions/activation_comparison.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def load_analysis_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return torch.load(path, map_location="cpu")

def compute_mean_activations(analysis_results, aggregation_method="mean_all"):
    """
    Computes mean activation vectors per layer for a given aggregation method.
    Returns a dictionary mapping layer -> mean activation vector.
    """
    aggregator_outputs = analysis_results.get("aggregator_outputs", {})
    mean_activations = {}
    for layer, methods in aggregator_outputs.items():
        data = methods.get(aggregation_method, None)
        if data is not None and data.size > 0:
            mean_activations[layer] = np.mean(data, axis=0)
        else:
            mean_activations[layer] = None
    return mean_activations

def compare_mean_activations(result_paths, aggregation_method="mean_all"):
    """
    Given a dictionary mapping prompt type to analysis file paths, compute the mean activations
    and then the pairwise cosine similarity for each layer.
    
    Returns:
       - A dict mapping prompt type -> { layer -> mean activation vector }
       - A dict mapping layer -> (list of prompt names, cosine similarity matrix)
    """
    prompt_means = {}
    for prompt, path in result_paths.items():
        results = load_analysis_results(path)
        prompt_means[prompt] = compute_mean_activations(results, aggregation_method=aggregation_method)
    
    # Determine common layers
    layers = None
    for means in prompt_means.values():
        if layers is None:
            layers = set(means.keys())
        else:
            layers &= set(means.keys())
    layers = sorted(layers, key=lambda x: int(x.split('_')[1]) if isinstance(x, str) and x.startswith("layer_") else int(x))
    
    # For each layer, compute pairwise cosine similarity between prompt types
    layer_similarities = {}
    prompt_names = list(prompt_means.keys())
    for layer in layers:
        vectors = []
        valid_prompts = []
        for prompt in prompt_names:
            vec = prompt_means[prompt].get(layer, None)
            if vec is not None:
                vectors.append(vec)
                valid_prompts.append(prompt)
        if len(vectors) > 1:
            vectors = np.stack(vectors)
            sim_matrix = cosine_similarity(vectors)
            layer_similarities[layer] = (valid_prompts, sim_matrix)
        else:
            layer_similarities[layer] = (valid_prompts, None)
    return prompt_means, layer_similarities

def plot_similarity_heatmap(sim_matrix, labels, title="Cosine Similarity Heatmap"):
    """
    Plots a heatmap of the cosine similarity matrix using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(sim_matrix, interpolation='nearest', cmap='viridis')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    plt.title(title)
    fig.colorbar(cax)
    plt.tight_layout()
    plt.show()
