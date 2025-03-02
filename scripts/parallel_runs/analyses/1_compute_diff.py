import os
import torch
import glob
from tqdm import tqdm

# Directories: Combined activation files and output differences
extractions_dir = globals().get("EXTRACTIONS_DIR", "output/extractions")
diff_dir = globals().get("DIFF_DIR", "output/differences")
os.makedirs(diff_dir, exist_ok=True)

# List all activation files (e.g., activations_*.pt)
files = sorted(glob.glob(os.path.join(extractions_dir, "*.pt")))
print(f"Found {len(files)} files in {extractions_dir}")

for file in tqdm(files, desc="Processing extraction files"):
    # Load the combined file of both "neutral" and "jb" keys
    data = torch.load(file, map_location="cpu")
    diff_data = {}

    # Process hidden_states: stored as dictionaries keyed by layer
    if "neutral" in data and "hidden_states" in data["neutral"] and "hidden_states" in data["jb"]:
        diff_data["hidden_states"] = {}
        for layer_key in data["neutral"]["hidden_states"]:
            neutral_tensor = data["neutral"]["hidden_states"][layer_key]
            jb_tensor = data["jb"]["hidden_states"].get(layer_key)
            if jb_tensor is None:
                continue
            # Crop along the token dimension.
            seq_len = min(neutral_tensor.size(0), jb_tensor.size(0))
            diff_data["hidden_states"][layer_key] = neutral_tensor[:seq_len] - jb_tensor[:seq_len]

    # Process attention_scores: cropping along both dimensions
    if "neutral" in data and "attention_scores" in data["neutral"] and "attention_scores" in data["jb"]:
        diff_data["attention_scores"] = {}
        for layer_key in data["neutral"]["attention_scores"]:
            neutral_tensor = data["neutral"]["attention_scores"][layer_key]
            jb_tensor = data["jb"]["attention_scores"].get(layer_key)
            if jb_tensor is None:
                continue
            # Crop along both dimensions (e.g., token dimension and attention dimension)
            min_dim0 = min(neutral_tensor.size(0), jb_tensor.size(0))
            min_dim1 = min(neutral_tensor.size(1), jb_tensor.size(1))
            diff_data["attention_scores"][layer_key] = neutral_tensor[:min_dim0, :min_dim1] - jb_tensor[:min_dim0, :min_dim1]

    # Process top_k_logits: stored as dictionaries keyed by token index
    if "neutral" in data and "top_k_logits" in data["neutral"] and "top_k_logits" in data["jb"]:
        diff_data["top_k_logits"] = {}
        for token_key in data["neutral"]["top_k_logits"]:
            neutral_tensor = data["neutral"]["top_k_logits"][token_key]
            jb_tensor = data["jb"]["top_k_logits"].get(token_key)
            if jb_tensor is None:
                continue
            diff_data["top_k_logits"][token_key] = neutral_tensor - jb_tensor

    # Process top_k_probs similarly
    if "neutral" in data and "top_k_probs" in data["neutral"] and "top_k_probs" in data["jb"]:
        diff_data["top_k_probs"] = {}
        for token_key in data["neutral"]["top_k_probs"]:
            neutral_tensor = data["neutral"]["top_k_probs"][token_key]
            jb_tensor = data["jb"]["top_k_probs"].get(token_key)
            if jb_tensor is None:
                continue
            diff_data["top_k_probs"][token_key] = neutral_tensor - jb_tensor

    # Save the computed differences for this file
    base_filename = os.path.basename(file)
    diff_output_path = os.path.join(diff_dir, base_filename)
    torch.save(diff_data, diff_output_path)
