import os
import sys
import logging
import pandas as pd
import glob

##############################################################################
# 1. Configuration
##############################################################################
# The directory where find_neurons_diff.py wrote its CSVs
OUT_DIR = (
    os.environ.get("OUT_DIR")
    or globals().get("OUT_DIR", "analysis/gemma2b/good_jb_neuron")
)
# The layers for which CSVs exist
LAYER_LIST = (
    os.environ.get("LAYER_LIST")
    or globals().get("LAYER_LIST", "0,5,10,15,20,25")
)
if isinstance(LAYER_LIST, str):
    LAYER_LIST = [int(x.strip()) for x in LAYER_LIST.split(",")]

# If you want to pick top K across the entire model, set TOP_K here
TOP_K_GLOBAL = int(
    os.environ.get("TOP_K_GLOBAL")
    or globals().get("TOP_K_GLOBAL", 50)
)

LOG_FILE = (
    os.environ.get("LOG_FILE")
    or globals().get("LOG_FILE", "logs/interpret_find_neurons_diff.log")
)

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("InterpretDiff")

print(f"=== Starting interpret_find_neurons_diff ===")
print(f"OUT_DIR = {OUT_DIR}")
print(f"LAYER_LIST = {LAYER_LIST}")
print(f"TOP_K_GLOBAL = {TOP_K_GLOBAL}")

##############################################################################
# 2. Summarize each layer's CSV
##############################################################################

def summarize_layer_csv(layer_csv):
    """
    Reads a CSV with columns:
      neuron_idx, safe_mean, harm_mean, diff_harm_minus_safe, abs_diff
    Returns:
      df (pd.DataFrame) fully loaded
      summary (str) text summary
    """
    df = pd.read_csv(layer_csv)
    # Basic stats
    count = len(df)
    max_abs_diff = df["abs_diff"].max()
    min_abs_diff = df["abs_diff"].min()
    mean_abs_diff = df["abs_diff"].mean()

    # The top row
    max_row = df.loc[df["abs_diff"].idxmax()]

    summary = (
        f"\n--- {os.path.basename(layer_csv)} ---\n"
        f"Total neurons: {count}\n"
        f"abs_diff range: [{min_abs_diff:.3f}, {max_abs_diff:.3f}], mean={mean_abs_diff:.3f}\n"
        f"Neuron with max abs_diff => idx={int(max_row['neuron_idx'])}, "
        f"safe_mean={max_row['safe_mean']:.3f}, harm_mean={max_row['harm_mean']:.3f}, "
        f"abs_diff={max_row['abs_diff']:.3f}\n"
    )
    return df, summary

##############################################################################
# 3. Load each layer, produce summary
##############################################################################
all_rows = []
for layer_id in LAYER_LIST:
    layer_csv = os.path.join(OUT_DIR, f"layer_{layer_id}_neuron_diff.csv")
    if not os.path.isfile(layer_csv):
        print(f"[WARN] Missing CSV for layer {layer_id}: {layer_csv}")
        continue

    df_layer, summ = summarize_layer_csv(layer_csv)
    print(summ)
    logger.info(summ)

    # We also keep these rows to do a global "top-K across all layers"
    df_layer["layer"] = layer_id
    all_rows.append(df_layer)

# Combine all layers if you want a single DataFrame
if all_rows:
    df_all = pd.concat(all_rows, axis=0, ignore_index=True)
    # Sorting by abs_diff descending
    df_all_sorted = df_all.sort_values("abs_diff", ascending=False)

    # If you want the top-K across the entire model:
    top_global = df_all_sorted.head(TOP_K_GLOBAL)
    out_csv_global = os.path.join(OUT_DIR, f"top_{TOP_K_GLOBAL}_neurons_GLOBAL.csv")
    top_global.to_csv(out_csv_global, index=False)
    msg = (
        f"\n=== Overall top {TOP_K_GLOBAL} neurons across all layers ===\n"
        f"Wrote to {out_csv_global}\n"
    )
    print(msg)
    logger.info(msg)

    # Optionally, you can do e.g. top_global[[...some columns...]].head(10) to show a snippet
    # or do further analysis. 
else:
    print("[ERROR] No data loaded, maybe none of the expected CSVs were found.")


print("=== Done interpreting find_neurons_diff output ===")
