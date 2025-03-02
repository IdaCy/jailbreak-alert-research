import os
import torch
import matplotlib.pyplot as plt

# (Optional) If you want to silence the PyTorch warning about pickle:
torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])

# Paths for the PCA results
PCA_DIR      = globals().get("PCA_DIR", "outputs/PCA")
RESULTS_PT   = globals().get("RESULTS_PT", os.path.join(PCA_DIR, "layer_pca_results.pt"))
PC1_FILE     = globals().get("PC1_FILE", os.path.join(PCA_DIR, "layer_pc1_vectors.pt"))

# Check that the files exist
if not os.path.exists(RESULTS_PT):
    raise ValueError(f"PCA results file not found: {RESULTS_PT}")
if not os.path.exists(PC1_FILE):
    raise ValueError(f"PC1 vectors file not found: {PC1_FILE}")

# Load the PCA results
pca_results = torch.load(RESULTS_PT, map_location="cpu")
pc1_vectors = torch.load(PC1_FILE, map_location="cpu")

print("=== PCA Results Summary ===")
print("Type of pca_results:", type(pca_results))
print("Type of pc1_vectors:", type(pc1_vectors))
print(f"Total layers: {len(pca_results)}\n")

# Sort layer keys by their numeric part
sorted_layers = sorted(pca_results.keys(), key=lambda x: int(x.split('_')[1]))
for layer in sorted_layers:
    ev_ratios = pca_results[layer]
    pc1_vec = pc1_vectors[layer]
    print(f"Layer {layer}: Explained variance ratio (top 3): {ev_ratios[:3]}")
    print(f"  PC1 first 3 elements: {pc1_vec[:3]}")
    print("")

# Plot the explained variance ratio of PC1 across layers
numeric_layers = [int(k.split("_")[1]) for k in sorted_layers]
first_pc_ev = [pca_results[layer][0] for layer in sorted_layers]

plt.figure(figsize=(10, 5))
plt.plot(numeric_layers, first_pc_ev, marker='o', linestyle='-')
plt.xlabel("Layer")
plt.ylabel("Explained Variance Ratio (PC1)")
plt.title("PC1 Explained Variance Ratio per Layer")
plt.grid(True)

plt.tight_layout()
plt.show()
