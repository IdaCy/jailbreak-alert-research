import os
import glob
import torch
from tqdm import tqdm

# Directories containing the separate .pt files from your new inference script
NEUTRAL_DIR = globals().get("NEUTRAL_DIR", "outputs/extractions/jailbreak_neutral")
JB_DIR      = globals().get("JB_DIR", "outputs/extractions/jailbreak")
DIFF_DIR    = globals().get("DIFF_DIR", "outputs/differences")
os.makedirs(DIFF_DIR, exist_ok=True)

# List all NEUTRAL .pt files
neutral_files = sorted(glob.glob(os.path.join(NEUTRAL_DIR, "*.pt")))
jb_files      = sorted(glob.glob(os.path.join(JB_DIR,      "*.pt")))

if len(neutral_files) != len(jb_files):
    raise ValueError("Mismatch: # of neutral files != # of jb files. Ensure same # of .pt files.")

print(f"Found {len(neutral_files)} neutral files in {NEUTRAL_DIR}")
print(f"Found {len(jb_files)} jb files in {JB_DIR}")

for neutral_file, jb_file in tqdm(zip(neutral_files, jb_files),
                                  desc="Computing differences",
                                  total=len(neutral_files)):

    # Load the .pt data from both sets
    neutral_data = torch.load(neutral_file, map_location="cpu")
    jb_data      = torch.load(jb_file,      map_location="cpu")

    diff_data = {}

    # ----------------------------------------------------------------
    # 1) Hidden states difference
    #    In your new inference script, "hidden_states" is a dict:
    #       { "layer_5": [batch, seq_len, hidden_dim], ... }
    # ----------------------------------------------------------------
    if "hidden_states" in neutral_data and "hidden_states" in jb_data:
        diff_data["hidden_states"] = {}
        for layer_key, neutral_tensor in neutral_data["hidden_states"].items():
            if layer_key in jb_data["hidden_states"]:
                jb_tensor = jb_data["hidden_states"][layer_key]
                # Both are shape [batch, seq_len, hidden_dim]
                # Crop the seq dimension if needed
                min_batch  = min(neutral_tensor.size(0), jb_tensor.size(0))
                min_seq    = min(neutral_tensor.size(1), jb_tensor.size(1))
                diff_data["hidden_states"][layer_key] = \
                    neutral_tensor[:min_batch, :min_seq] - jb_tensor[:min_batch, :min_seq]

    # ----------------------------------------------------------------
    # 2) Attentions difference (optional)
    #    In your new script, this is "attentions" not "attention_scores".
    #    The shape: [batch, num_heads, seq_len, seq_len]
    # ----------------------------------------------------------------
    if "attentions" in neutral_data and "attentions" in jb_data:
        diff_data["attentions"] = {}
        for layer_key, neutral_tensor in neutral_data["attentions"].items():
            if layer_key in jb_data["attentions"]:
                jb_tensor = jb_data["attentions"][layer_key]
                min_batch = min(neutral_tensor.size(0), jb_tensor.size(0))
                min_heads = min(neutral_tensor.size(1), jb_tensor.size(1))
                min_seq   = min(neutral_tensor.size(2), jb_tensor.size(2))
                diff_data["attentions"][layer_key] = (
                    neutral_tensor[:min_batch, :min_heads, :min_seq, :min_seq]
                    - jb_tensor[:min_batch, :min_heads, :min_seq, :min_seq]
                )

    # ----------------------------------------------------------------
    # 3) Top-K logits difference (optional)
    #    The new script calls it "topk_logits" (not "top_k_logits").
    #    Shape: [batch, seq_len, top_k]
    # ----------------------------------------------------------------
    if "topk_logits" in neutral_data and "topk_logits" in jb_data:
        # We'll store one key: "topk_logits"
        neutral_topk = neutral_data["topk_logits"]
        jb_topk      = jb_data["topk_logits"]
        min_batch    = min(neutral_topk.size(0), jb_topk.size(0))
        min_seq      = min(neutral_topk.size(1), jb_topk.size(1))
        diff_data["topk_logits"] = (
            neutral_topk[:min_batch, :min_seq] - jb_topk[:min_batch, :min_seq]
        )

    # ----------------------------------------------------------------
    # 4) Save the computed difference
    #    We'll base the filename on the neutral file’s name
    # ----------------------------------------------------------------
    base_filename = os.path.basename(neutral_file)
    diff_path = os.path.join(DIFF_DIR, base_filename)
    torch.save(diff_data, diff_path)
