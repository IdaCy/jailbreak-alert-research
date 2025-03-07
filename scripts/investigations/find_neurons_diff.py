import os
import glob
import torch
import logging
import math
import numpy as np
import pandas as pd

##############################################################################
# 1. Config / Env Variables
##############################################################################
SAFE_DIR = (
    os.environ.get("SAFE_DIR")
    or globals().get("SAFE_DIR", "output/extractions/gemma2b/good")
)

HARM_DIR = (
    os.environ.get("HARM_DIR")
    or globals().get("HARM_DIR", "output/extractions/gemma2b/jb")
)
OUT_DIR = (
    os.environ.get("OUT_DIR")
    or globals().get("OUT_DIR", "analysis/gemma2b/good_jb_neuron")
)
os.makedirs(OUT_DIR, exist_ok=True)

# e.g. "0,5,10,15,20,25" => [0,5,10,15,20,25]
EXTRACT_HIDDEN_LAYERS = (
    os.environ.get("EXTRACT_HIDDEN_LAYERS")
    or globals().get("EXTRACT_HIDDEN_LAYERS", "0,5,10,15,20,25")
)
if isinstance(EXTRACT_HIDDEN_LAYERS, str):
    EXTRACT_HIDDEN_LAYERS = [int(x.strip()) for x in EXTRACT_HIDDEN_LAYERS.split(",")]

# how many neurons do we print at the top of each layer
TOP_K = int(
    os.environ.get("TOP_K")
    or globals().get("TOP_K", 50)
)

LOG_FILE = os.path.join(OUT_DIR, "find_neurons_diff.log")

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

logger = logging.getLogger("NeuronDiff")

logger.info("=== Starting neuron-difference script ===")
logger.info(f"SAFE_DIR = {SAFE_DIR}")
logger.info(f"HARM_DIR = {HARM_DIR}")
logger.info(f"OUT_DIR  = {OUT_DIR}")
logger.info(f"Layers   = {EXTRACT_HIDDEN_LAYERS}")
logger.info(f"TOP_K    = {TOP_K}")

##############################################################################
# 2. Pair up .pt files by matching filenames
##############################################################################
def pair_files(safe_dir, harm_dir):
    safe_files = sorted(glob.glob(os.path.join(safe_dir, "activations_*.pt")))
    harm_files = sorted(glob.glob(os.path.join(harm_dir, "activations_*.pt")))

    # map basename -> full path
    safe_map = {os.path.basename(fp): fp for fp in safe_files}
    harm_map = {os.path.basename(fp): fp for fp in harm_files}

    # intersection
    matched = []
    for base in safe_map:
        if base in harm_map:
            matched.append((safe_map[base], harm_map[base]))

    matched.sort(key=lambda x: x[0])
    logger.info(f"Found {len(safe_files)} safe files, {len(harm_files)} harmful files.")
    logger.info(f"Paired up {len(matched)} matching filenames.")
    return matched

##############################################################################
# 3. Main code to load & gather stats
##############################################################################
def main():
    pairs = pair_files(SAFE_DIR, HARM_DIR)
    if not pairs:
        logger.error("No matched files found between safe/harmful dirs.")
        return

    # We'll accumulate sums of activations and counts for each layer, for safe and harmful separately.
    # layer_stats[l]["safe_sum"]   => shape [hidden_dim]
    # layer_stats[l]["safe_count"] => int
    # layer_stats[l]["harm_sum"]   => shape [hidden_dim]
    # layer_stats[l]["harm_count"] => int
    layer_stats = {}
    for l in EXTRACT_HIDDEN_LAYERS:
        layer_stats[l] = {
            "safe_sum": None,
            "safe_count": 0,
            "harm_sum": None,
            "harm_count": 0
        }

    # 3.1. Iterate over pairs
    for (safe_fp, harm_fp) in pairs:
        try:
            safe_data = torch.load(safe_fp)
            harm_data = torch.load(harm_fp)

            # hidden_states -> { "layer_0": shape [batch_size, seq_len, hidden_dim], ... }
            safe_hs = safe_data["hidden_states"]
            harm_hs = harm_data["hidden_states"]

            for l in EXTRACT_HIDDEN_LAYERS:
                key = f"layer_{l}"
                if key not in safe_hs or key not in harm_hs:
                    continue

                safe_tensor = safe_hs[key]  # shape [batch_size, seq_len, hidden_dim]
                harm_tensor = harm_hs[key]

                # We'll flatten across batch_size * seq_len -> shape [N, hidden_dim].
                # This lumps all tokens together so we can compute average.
                safe_np = safe_tensor.to(torch.float32).view(-1, safe_tensor.shape[-1]).cpu().numpy()
                harm_np = harm_tensor.to(torch.float32).view(-1, harm_tensor.shape[-1]).cpu().numpy()

                if layer_stats[l]["safe_sum"] is None:
                    layer_stats[l]["safe_sum"] = safe_np.sum(axis=0)
                    layer_stats[l]["safe_count"] = safe_np.shape[0]
                else:
                    layer_stats[l]["safe_sum"] += safe_np.sum(axis=0)
                    layer_stats[l]["safe_count"] += safe_np.shape[0]

                if layer_stats[l]["harm_sum"] is None:
                    layer_stats[l]["harm_sum"] = harm_np.sum(axis=0)
                    layer_stats[l]["harm_count"] = harm_np.shape[0]
                else:
                    layer_stats[l]["harm_sum"] += harm_np.sum(axis=0)
                    layer_stats[l]["harm_count"] += harm_np.shape[0]

        except Exception as e:
            logger.error(f"Error loading or processing: {safe_fp}, {harm_fp}, err={e}")

    # 3.2. Now compute mean activation, difference
    # We'll produce a CSV for each layer with columns:
    # neuron_index, safe_mean, harm_mean, diff(harm-safe), abs_diff
    for l in EXTRACT_HIDDEN_LAYERS:
        info = layer_stats[l]
        if (info["safe_sum"] is None) or (info["harm_sum"] is None):
            logger.warning(f"No data found for layer {l}, skipping.")
            continue

        safe_mean = info["safe_sum"] / max(info["safe_count"], 1)
        harm_mean = info["harm_sum"] / max(info["harm_count"], 1)

        diff = harm_mean - safe_mean  # shape [hidden_dim]
        abs_diff = np.abs(diff)
        hidden_dim = diff.shape[0]

        # Build a DataFrame of results
        df = pd.DataFrame({
            "neuron_idx": np.arange(hidden_dim),
            "safe_mean": safe_mean,
            "harm_mean": harm_mean,
            "diff_harm_minus_safe": diff,
            "abs_diff": abs_diff
        })

        # Sort by abs_diff descending
        df_sorted = df.sort_values("abs_diff", ascending=False).reset_index(drop=True)

        # Save full CSV
        out_csv = os.path.join(OUT_DIR, f"layer_{l}_neuron_diff.csv")
        df_sorted.to_csv(out_csv, index=False)
        logger.info(f"Wrote per-neuron difference results for layer {l} to {out_csv}")

        # Print top-K
        logger.info(f"Top {TOP_K} neurons for layer {l} with biggest abs_diff:")
        logger.info("\n" + df_sorted.head(TOP_K).to_string(index=False))

if __name__ == "__main__":
    main()
