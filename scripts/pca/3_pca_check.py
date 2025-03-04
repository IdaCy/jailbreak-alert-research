import os
import torch
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Set up logging
LOG_DIR = "logs"
log_filename = os.path.join(LOG_DIR, "3_pca_check.log")
#timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#log_filename = os.path.join(LOG_DIR, f"log_{timestamp}.txt")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# (Optional) If you want to silence the PyTorch warning about pickle:
torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])

# Paths for the PCA results
PCA_DIR = (
    os.environ.get("PCA_DIR")
    or globals().get("PCA_DIR")
    or "outputs/PCA"
)

RESULTS_PT = (
    os.environ.get("RESULTS_PT")
    or globals().get("RESULTS_PT")
    or os.path.join(PCA_DIR, "layer_pca_results.pt")
)

PC1_FILE = (
    os.environ.get("PC1_FILE")
    or globals().get("PC1_FILE")
    or os.path.join(PCA_DIR, "layer_pc1_vectors.pt")
)

# Check that the files exist
if not os.path.exists(RESULTS_PT):
    msg = f"PCA results file not found: {RESULTS_PT}"
    logging.error(msg)
    raise ValueError(msg)
else:
    logging.info(f"Found PCA results file: {RESULTS_PT}")

if not os.path.exists(PC1_FILE):
    msg = f"PC1 vectors file not found: {PC1_FILE}"
    logging.error(msg)
    raise ValueError(msg)
else:
    logging.info(f"Found PC1 vectors file: {PC1_FILE}")

# Load the PCA results
pca_results = torch.load(RESULTS_PT, map_location="cpu")
pc1_vectors = torch.load(PC1_FILE, map_location="cpu")

print("=== PCA Results Summary ===")
print("Type of pca_results:", type(pca_results))
print("Type of pc1_vectors:", type(pc1_vectors))
print(f"Total layers: {len(pca_results)}\n")

logging.info("=== PCA Results Summary ===")
logging.info(f"Type of pca_results: {type(pca_results)}")
logging.info(f"Type of pc1_vectors: {type(pc1_vectors)}")
logging.info(f"Total layers: {len(pca_results)}")

# Sort layer keys by their numeric part
sorted_layers = sorted(pca_results.keys(), key=lambda x: int(x.split('_')[1]))

for layer in sorted_layers:
    ev_ratios = pca_results[layer]
    pc1_vec = pc1_vectors[layer]
    print(f"Layer {layer}: Explained variance ratio (top 3): {ev_ratios[:3]}")
    print(f"  PC1 first 3 elements: {pc1_vec[:3]}")
    print("")

    logging.info(f"Layer {layer}: Explained variance ratio (top 3): {ev_ratios[:3]}")
    logging.info(f"  PC1 first 3 elements: {pc1_vec[:3]}")

# Plot the explained variance ratio of PC1 across layers
numeric_layers = [int(k.split("_")[1]) for k in sorted_layers]
first_pc_ev = [pca_results[layer][0] for layer in sorted_layers]

logging.info("Plotting PC1 explained variance ratio per layer.")

plt.figure(figsize=(10, 5))
plt.plot(numeric_layers, first_pc_ev, marker='o', linestyle='-')
plt.xlabel("Layer")
plt.ylabel("Explained Variance Ratio (PC1)")
plt.title("PC1 Explained Variance Ratio per Layer")
plt.grid(True)

plt.tight_layout()
plt.show()

logging.info("Plot successfully displayed.")
