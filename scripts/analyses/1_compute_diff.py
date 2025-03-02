import os
import torch
import glob
from tqdm import tqdm

# Directories: Combined activation files and output differences
extractions_dir = globals().get("EXTRACTIONS_DIR", "data/extractions")
diff_dir = globals().get("DIFF_DIR", "output/differences")
os.makedirs(diff_dir, exist_ok=True)

# List all activation files (e.g., activations_*.pt)
files = sorted(glob.glob(os.path.join(extractions_dir, "*.pt")))
print(f"Found {len(files)} files in {extractions_dir}")

for file in tqdm(files, desc="Processing extraction files"):
    # Load the combined file of both "clean" and "typo" keys
    data = torch.load(file, map_location="cpu")
    diff_data = {}

    # Process hidden_states: stored as dictionaries keyed by layer
    if "clean" in data and "hidden_states" in data["clean"] and "hidden_states" in data["typo"]:
        diff_data["hidden_states"] = {}
        for layer_key in data["clean"]["hidden_states"]:
            clean_tensor = data["clean"]["hidden_states"][layer_key]
            typo_tensor = data["typo"]["hidden_states"].get(layer_key)
            if typo_tensor is None:
                continue
            # Crop along the token dimension.
            seq_len = min(clean_tensor.size(0), typo_tensor.size(0))
            diff_data["hidden_states"][layer_key] = clean_tensor[:seq_len] - typo_tensor[:seq_len]

    # Process attention_scores: cropping along both dimensions
    if "clean" in data and "attention_scores" in data["clean"] and "attention_scores" in data["typo"]:
        diff_data["attention_scores"] = {}
        for layer_key in data["clean"]["attention_scores"]:
            clean_tensor = data["clean"]["attention_scores"][layer_key]
            typo_tensor = data["typo"]["attention_scores"].get(layer_key)
            if typo_tensor is None:
                continue
            # Crop along both dimensions (e.g., token dimension and attention dimension)
            min_dim0 = min(clean_tensor.size(0), typo_tensor.size(0))
            min_dim1 = min(clean_tensor.size(1), typo_tensor.size(1))
            diff_data["attention_scores"][layer_key] = clean_tensor[:min_dim0, :min_dim1] - typo_tensor[:min_dim0, :min_dim1]

    # Process top_k_logits: stored as dictionaries keyed by token index
    if "clean" in data and "top_k_logits" in data["clean"] and "top_k_logits" in data["typo"]:
        diff_data["top_k_logits"] = {}
        for token_key in data["clean"]["top_k_logits"]:
            clean_tensor = data["clean"]["top_k_logits"][token_key]
            typo_tensor = data["typo"]["top_k_logits"].get(token_key)
            if typo_tensor is None:
                continue
            diff_data["top_k_logits"][token_key] = clean_tensor - typo_tensor

    # Process top_k_probs similarly
    if "clean" in data and "top_k_probs" in data["clean"] and "top_k_probs" in data["typo"]:
        diff_data["top_k_probs"] = {}
        for token_key in data["clean"]["top_k_probs"]:
            clean_tensor = data["clean"]["top_k_probs"][token_key]
            typo_tensor = data["typo"]["top_k_probs"].get(token_key)
            if typo_tensor is None:
                continue
            diff_data["top_k_probs"][token_key] = clean_tensor - typo_tensor

    # Save the computed differences for this file
    base_filename = os.path.basename(file)
    diff_output_path = os.path.join(diff_dir, base_filename)
    torch.save(diff_data, diff_output_path)
