import os
import torch
import glob
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

# Set up logging
LOG_DIR = "logs"
log_filename = os.path.join(LOG_DIR, "2_run_pca.log")
#timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#log_filename = os.path.join(LOG_DIR, f"log_{timestamp}.txt")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Directories for input differences and output analyses.
DIFF_DIR = (
    os.environ.get("DIFF_DIR")
    or globals().get("DIFF_DIR")
    or "outputs/differences"
)

PCA_DIR = (
    os.environ.get("PCA_DIR")
    or globals().get("PCA_DIR")
    or "outputs/PCA"
)
os.makedirs(PCA_DIR, exist_ok=True)

diff_files = sorted(glob.glob(os.path.join(DIFF_DIR, "*.pt")))
print("Total difference files found:", len(diff_files))
logging.info(f"Total difference files found: {len(diff_files)}")

if len(diff_files) == 0:
    msg = f"No difference files found in directory: {DIFF_DIR}"
    logging.error(msg)
    raise ValueError(msg)

# Load one sample file to see what keys exist
sample_data = torch.load(diff_files[0], map_location="cpu")

# We expect a "hidden_states" dict like: { "layer_0": tensor, "layer_5": tensor, ... }
if isinstance(sample_data, dict) and "hidden_states" in sample_data:
    layer_keys = sorted(sample_data["hidden_states"].keys(), key=lambda x: int(x.split("_")[1]))
else:
    msg = "Unexpected format in difference file. Expected dict with 'hidden_states'."
    logging.error(msg)
    raise ValueError(msg)

print("Layer keys detected:", layer_keys)
logging.info(f"Layer keys detected: {layer_keys}")

# We'll store PCA results (explained variance ratios) and the PC1 vector for each layer
layer_pca_results = {}
layer_pc1_vectors = {}

# Set maximum number of vectors per layer to keep in memory
max_samples = 10000
num_workers = 8  # For ThreadPoolExecutor

def process_file_for_layer(file, layer_key):
    """
    Returns flattened difference vectors for the specified layer key
    from one difference file.
    """
    try:
        diff_data = torch.load(file, map_location="cpu")
        if "hidden_states" in diff_data and layer_key in diff_data["hidden_states"]:
            tensor = diff_data["hidden_states"][layer_key]  # shape [batch, seq_len, hidden_dim]
            # Convert from bfloat16 -> float32
            tensor = tensor.to(torch.float32)
            flat = tensor.reshape(-1, tensor.shape[-1])
            return flat.numpy()
    except Exception as e:
        logging.error(f"Error processing {file} for {layer_key}: {e}")
        print(f"Error processing {file} for {layer_key}: {e}")
    return None

for layer_key in layer_keys:
    print(f"\nProcessing {layer_key}...")
    logging.info(f"Processing {layer_key}...")
    all_diffs = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file_for_layer, f, layer_key): f for f in diff_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{layer_key} files"):
            result = fut.result()
            if result is not None:
                all_diffs.append(result)

    if len(all_diffs) == 0:
        print(f"No data collected for {layer_key}. Skipping.")
        logging.warning(f"No data collected for {layer_key}. Skipping.")
        continue

    # Concatenate all arrays from this layer
    all_diff_vectors = np.concatenate(all_diffs, axis=0)
    print(f"Collected {all_diff_vectors.shape[0]} difference vectors for {layer_key}.")
    logging.info(f"Collected {all_diff_vectors.shape[0]} difference vectors for {layer_key}.")

    # Subsample if needed
    if all_diff_vectors.shape[0] > max_samples:
        indices = np.random.choice(all_diff_vectors.shape[0], size=max_samples, replace=False)
        all_diff_vectors = all_diff_vectors[indices]
        print(f"Subsampled to {max_samples} vectors for {layer_key}.")
        logging.info(f"Subsampled to {max_samples} vectors for {layer_key}.")

    # Run PCA
    pca = PCA(n_components=10)
    pca.fit(all_diff_vectors)
    explained_variance = pca.explained_variance_ratio_
    layer_pca_results[layer_key] = explained_variance

    # Save the first principal component
    pc1 = pca.components_[0]  # shape [hidden_dim]
    layer_pc1_vectors[layer_key] = pc1
    print(f"{layer_key}: Top 10 explained variance ratios: {explained_variance}")
    logging.info(f"{layer_key}: Top 10 explained variance ratios: {explained_variance}")

# Save results
results_file = os.path.join(PCA_DIR, "layer_pca_results.pt")
torch.save(layer_pca_results, results_file)
print(f"PCA results saved to {results_file}")
logging.info(f"PCA results saved to {results_file}")

pc1_file = os.path.join(PCA_DIR, "layer_pc1_vectors.pt")
torch.save(layer_pc1_vectors, pc1_file)
print(f"PC1 vectors saved to {pc1_file}")
logging.info(f"PC1 vectors saved to {pc1_file}")

# Plot the variance of PC1 across layers
sorted_layers = sorted(layer_pca_results.keys(), key=lambda x: int(x.split("_")[1]))
pc1_variances = [layer_pca_results[lyr][0] for lyr in sorted_layers]

plt.figure(figsize=(10, 5))
plt.plot(sorted_layers, pc1_variances, marker='o')
plt.xlabel("Layer")
plt.ylabel("Explained Variance Ratio (PC1)")
plt.title("PC1 Explained Variance Ratio per Layer")
plt.grid(True)
plot_file = os.path.join(PCA_DIR, "pca_plot.png")
plt.savefig(plot_file)
plt.show()
plt.close()

print(f"PCA plot saved to {plot_file}")
logging.info(f"PCA plot saved to {plot_file}")
