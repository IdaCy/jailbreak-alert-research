import os
import torch
import glob
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Directories for input differences and output analyses
diff_dir = "analyses/differences"
output_dir = "analyses/PCA_results"

# List all difference files (they are in .pt format)
diff_files = sorted(glob.glob(os.path.join(diff_dir, "*.pt")))
print("Total difference files found:", len(diff_files))

# Determine the number of layers by loading one file
sample_data = torch.load(diff_files[0], map_location="cpu")
if isinstance(sample_data, dict) and "hidden_states" in sample_data:
    num_layers = len(sample_data["hidden_states"])
else:
    raise ValueError("Unexpected format in difference file. Expected dict with 'hidden_states'.")
print("Number of layers detected:", num_layers)

# We'll store PCA results (explained variance ratios) for each layer
layer_pca_results = {}

# Set maximum number of vectors per layer to use in PCA (to manage memory/computation)
max_samples = 10000
# Use 8 worker threads (matching your 8 cores)
num_workers = 8

def process_file_for_layer(file, layer):
    """
    Loads one difference file and returns the flattened difference vectors
    for the specified layer. If the file does not contain the requested layer,
    returns None.
    """
    try:
        diff_data = torch.load(file, map_location="cpu")
        if isinstance(diff_data, dict) and "hidden_states" in diff_data:
            hidden_states = diff_data["hidden_states"]
            if len(hidden_states) > layer:
                tensor = hidden_states[layer]  # shape: [batch, seq_len, hidden_dim]
                flat = tensor.reshape(-1, tensor.shape[-1])
                return flat.numpy()
    except Exception as e:
        print(f"Error processing {file} for layer {layer}: {e}")
    return None

for layer in range(num_layers):
    print(f"\nProcessing layer {layer}...")
    all_diff_vectors_list = []
    # Process files in parallel using a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file_for_layer, file, layer): file for file in diff_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Layer {layer} files"):
            result = future.result()
            if result is not None:
                all_diff_vectors_list.append(result)
    if len(all_diff_vectors_list) == 0:
        print(f"No data collected for layer {layer}.")
        continue
    # Concatenate all arrays from this layer
    all_diff_vectors = np.concatenate(all_diff_vectors_list, axis=0)
    print(f"Collected {all_diff_vectors.shape[0]} vectors for layer {layer}.")
    
    # Subsample if necessary to max_samples
    if all_diff_vectors.shape[0] > max_samples:
        indices = np.random.choice(all_diff_vectors.shape[0], size=max_samples, replace=False)
        all_diff_vectors = all_diff_vectors[indices]
        print(f"Subsampled to {max_samples} vectors for layer {layer}.")
    
    # Run PCA on the difference vectors for this layer
    pca = PCA(n_components=10)
    pca.fit(all_diff_vectors)
    explained_variance = pca.explained_variance_ratio_
    layer_pca_results[layer] = explained_variance
    print(f"Layer {layer}: Top 10 explained variance ratios: {explained_variance}")

# Save the PCA results dictionary to the output directory.
results_file = os.path.join(output_dir, "layer_pca_results.pt")
torch.save(layer_pca_results, results_file)
print(f"PCA results saved to {results_file}")

# Plot the explained variance ratio of the first principal component across layers
layers = sorted(layer_pca_results.keys())
first_pc = [layer_pca_results[layer][0] for layer in layers]

plt.figure(figsize=(10, 5))
plt.plot(layers, first_pc, marker='o')
plt.xlabel("Layer")
plt.ylabel("Explained Variance Ratio (PC1)")
plt.title("PC1 Explained Variance Ratio per Layer")
plt.grid(True)

# Save the plot to the output directory
plot_file = os.path.join(output_dir, "pca_plot.png")
plt.savefig(plot_file)
plt.close()  # close the plot to free memory
print(f"PCA plot saved to {plot_file}")
