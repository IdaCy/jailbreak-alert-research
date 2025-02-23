import os
import torch
import glob
from tqdm import tqdm

# Directories for activations
jailbreak_dir  = "data/jailbreak"
neutral_dir = "data/neutral"
diff_dir  = "analyses/differences"

# Create output directory for difference files if it doesn't exist.
os.makedirs(diff_dir, exist_ok=True)

# List and sort the files so they are paired correctly.
neutral_files = sorted(glob.glob(os.path.join(neutral_dir, "*.pt")))
jailbreak_files  = sorted(glob.glob(os.path.join(jailbreak_dir, "*.pt")))

assert len(neutral_files) == len(jailbreak_files), "Mismatch in number of neutral and jailbreak files."

for neutral_file in tqdm(neutral_files, desc="Processing activation files"):
    filename = os.path.basename(neutral_file)
    jailbreak_file = os.path.join(jailbreak_dir, filename)
    
    if not os.path.exists(jailbreak_file):
        print(f"jailbreak file {jailbreak_file} not found. Skipping.")
        continue

    # Load both neutral and jailbreak activation files on CPU.
    neutral_acts = torch.load(neutral_file, map_location="cpu")
    jailbreak_acts  = torch.load(jailbreak_file, map_location="cpu")
    
    diff_data = {}

    # Activations are stored as a dictionary with keys like:
    # 'hidden_states', 'input_ids', 'final_predictions'
    if isinstance(neutral_acts, dict):
        for key in neutral_acts.keys():
            if key not in jailbreak_acts:
                print(f"Key {key} not found in jailbreak file {filename}. Skipping this key.")
                continue

            if key == "hidden_states":
                # Expecting hidden_states to be a list of tensors, one per layer.
                neutral_hidden = neutral_acts[key]
                jailbreak_hidden = jailbreak_acts[key]
                diff_hidden = []
                for neutral_tensor, jailbreak_tensor in zip(neutral_hidden, jailbreak_hidden):
                    # Both tensors have shape [batch, seq_len, hidden_dim]
                    # But the sequence length may differ. Crop to the minimum.
                    seq_len = min(neutral_tensor.size(1), jailbreak_tensor.size(1))
                    diff_hidden.append(neutral_tensor[:, :seq_len, :] - jailbreak_tensor[:, :seq_len, :])
                diff_data[key] = diff_hidden
            else:
                # For keys like "input_ids" and "final_predictions", simply copy them.
                diff_data[key] = neutral_acts[key]
    elif isinstance(neutral_acts, list):
        # In case the activations are stored as a list rather than a dict.
        diff_data = []
        for i, act in enumerate(neutral_acts):
            # Convert lists to tensors if necessary.
            if isinstance(act, list):
                act = torch.tensor(act)
            jailbreak_act = jailbreak_acts[i]
            if isinstance(jailbreak_act, list):
                jailbreak_act = torch.tensor(jailbreak_act)
            if act.ndim >= 2 and jailbreak_act.ndim >= 2:
                # Crop to the minimum sequence length (sequence is first dim)
                seq_len = min(act.size(0), jailbreak_act.size(0))
                act = act[:seq_len]
                jailbreak_act = jailbreak_act[:seq_len]
            diff_data.append(act - jailbreak_act)
    else:
        raise ValueError("Unknown activation file format; expected dict or list.")

    # Save the computed differences for this file.
    diff_file = os.path.join(diff_dir, filename)
    torch.save(diff_data, diff_file)
