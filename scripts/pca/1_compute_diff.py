import os
import glob
import torch
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
LOG_DIR = "logs"
log_filename = os.path.join(LOG_DIR, "1_compute_diffs.log")
#timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#log_filename = os.path.join(LOG_DIR, f"log_{timestamp}.txt")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Directories containing the separate .pt files from your new inference script
JB_DIR = (
    os.environ.get("JB_DIR")
    or globals().get("JB_DIR")
    or "output/extractions/jb_small"
)

NEUTRAL_DIR = (
    os.environ.get("NEUTRAL_DIR")
    or globals().get("NEUTRAL_DIR")
    or "output/extractions/good"
)

DIFF_DIR = (
    os.environ.get("DIFF_DIR")
    or globals().get("DIFF_DIR")
    or "output/differences/jb_good_small"
)
os.makedirs(DIFF_DIR, exist_ok=True)

# List all NEUTRAL .pt files
neutral_files = sorted(glob.glob(os.path.join(NEUTRAL_DIR, "*.pt")))
jb_files = sorted(glob.glob(os.path.join(JB_DIR, "*.pt")))

if len(neutral_files) != len(jb_files):
    msg = (
        f"Mismatch: # of neutral files ({len(neutral_files)}) != "
        f"# of jb files ({len(jb_files)}). Ensure same number of .pt files."
    )
    logging.error(msg)
    raise ValueError(msg)

print(f"Found {len(neutral_files)} neutral files in {NEUTRAL_DIR}")
print(f"Found {len(jb_files)} jb files in {JB_DIR}")
logging.info(f"Found {len(neutral_files)} neutral files in {NEUTRAL_DIR}")
logging.info(f"Found {len(jb_files)} jb files in {JB_DIR}")

for neutral_file, jb_file in tqdm(
    zip(neutral_files, jb_files),
    desc="Computing differences",
    total=len(neutral_files)
):
    logging.info(f"Processing neutral file: {neutral_file}")
    logging.info(f"Processing JB file: {jb_file}")

    # Load the .pt data from both sets
    neutral_data = torch.load(neutral_file, map_location="cpu")
    jb_data = torch.load(jb_file, map_location="cpu")

    diff_data = {}

    # 1) Hidden states difference
    if "hidden_states" in neutral_data and "hidden_states" in jb_data:
        diff_data["hidden_states"] = {}
        for layer_key, neutral_tensor in neutral_data["hidden_states"].items():
            if layer_key in jb_data["hidden_states"]:
                jb_tensor = jb_data["hidden_states"][layer_key]
                # Both are shape [batch, seq_len, hidden_dim]
                # Crop the seq dimension if needed
                min_batch = min(neutral_tensor.size(0), jb_tensor.size(0))
                min_seq = min(neutral_tensor.size(1), jb_tensor.size(1))
                diff_data["hidden_states"][layer_key] = \
                    neutral_tensor[:min_batch, :min_seq] - jb_tensor[:min_batch, :min_seq]

    # 2) Attentions difference
    if "attentions" in neutral_data and "attentions" in jb_data:
        diff_data["attentions"] = {}
        for layer_key, neutral_tensor in neutral_data["attentions"].items():
            if layer_key in jb_data["attentions"]:
                jb_tensor = jb_data["attentions"][layer_key]
                min_batch = min(neutral_tensor.size(0), jb_tensor.size(0))
                min_heads = min(neutral_tensor.size(1), jb_tensor.size(1))
                min_seq = min(neutral_tensor.size(2), jb_tensor.size(2))
                diff_data["attentions"][layer_key] = (
                    neutral_tensor[:min_batch, :min_heads, :min_seq, :min_seq]
                    - jb_tensor[:min_batch, :min_heads, :min_seq, :min_seq]
                )

    # 3) Top-K logits difference
    if "topk_logits" in neutral_data and "topk_logits" in jb_data:
        neutral_topk = neutral_data["topk_logits"]
        jb_topk = jb_data["topk_logits"]
        min_batch = min(neutral_topk.size(0), jb_topk.size(0))
        min_seq = min(neutral_topk.size(1), jb_topk.size(1))
        diff_data["topk_logits"] = (
            neutral_topk[:min_batch, :min_seq] - jb_topk[:min_batch, :min_seq]
        )

    # 4) Save the computed difference
    base_filename = os.path.basename(neutral_file)
    diff_path = os.path.join(DIFF_DIR, base_filename)
    torch.save(diff_data, diff_path)
    logging.info(f"Saved difference data to {diff_path}")

print("All differences computed and saved.")
logging.info("All differences computed and saved.")
