import os
import glob
import torch
import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

###############################################################################
# 1. Configuration
###############################################################################
# Directories for the two runs:
GOOD_DIR = (
    os.environ.get("GOOD_DIR")
    or globals().get("GOOD_DIR")
    or "output/extractions/gemma2b/good"
)
JB_DIR = (
    os.environ.get("JB_DIR")
    or globals().get("JB_DIR")
    or "output/extractions/gemma2b/jb"
)

"""# Where we might store logs or final results
OUTPUT_DIR = (
    os.environ.get("OUTPUT_DIR")
    or globals().get("OUTPUT_DIR")
    or "output/extractions/gemma2b"
)"""

# Layers to extract
EXTRACT_HIDDEN_LAYERS = (
    os.environ.get("EXTRACT_HIDDEN_LAYERS")
    or globals().get("EXTRACT_HIDDEN_LAYERS", [0, 5, 10, 15, 20, 25])
)
if isinstance(EXTRACT_HIDDEN_LAYERS, str):
    EXTRACT_HIDDEN_LAYERS = [int(x.strip()) for x in EXTRACT_HIDDEN_LAYERS.split(",")]

# Train/test ratio
TRAIN_TEST_SPLIT_RATIO = float(
    os.environ.get("TRAIN_TEST_SPLIT_RATIO")
    or globals().get("TRAIN_TEST_SPLIT_RATIO", 0.8)
)

os.makedirs("logs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"logs/lin_class_{timestamp}.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

logging.info("Starting linear classification script to pair up 'good' vs 'jb' runs.")
logging.info(f"GOOD_DIR: {GOOD_DIR}")
logging.info(f"JB_DIR: {JB_DIR}")
#logging.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
logging.info(f"EXTRACT_HIDDEN_LAYERS: {EXTRACT_HIDDEN_LAYERS}")
logging.info(f"TRAIN_TEST_SPLIT_RATIO: {TRAIN_TEST_SPLIT_RATIO}")
logging.info(f"Logging to: {LOG_FILE}")


###############################################################################
# 2. Step: Pair up the .pt files from GOOD_DIR and JB_DIR
###############################################################################
def pair_activation_files(good_dir, jb_dir):
    """
    We look for files named 'activations_XXXXX_YYYYY.pt' in both directories.
    We'll match them by the exact filename, ignoring any that only exist in one place.

    Returns a list of tuples: [(good_file, jb_file), (good_file, jb_file), ...].
    """
    good_files = sorted(glob.glob(os.path.join(good_dir, "activations_*.pt")))
    jb_files   = sorted(glob.glob(os.path.join(jb_dir,   "activations_*.pt")))

    # Convert them to dictionaries: filename -> full path
    good_basename_map = {os.path.basename(fp): fp for fp in good_files}
    jb_basename_map   = {os.path.basename(fp): fp for fp in jb_files}

    # Intersection of basenames
    matched_pairs = []
    for base in good_basename_map.keys():
        if base in jb_basename_map:
            matched_pairs.append((good_basename_map[base], jb_basename_map[base]))

    matched_pairs.sort(key=lambda x: x[0])  # sort by the good_file path or something

    logging.info(f"Found {len(good_files)} good .pt files, {len(jb_files)} jb .pt files.")
    logging.info(f"Paired up {len(matched_pairs)} matching filenames.")
    return matched_pairs


###############################################################################
# 3. Load and Combine Activations
###############################################################################
def load_and_combine(good_jb_pairs):
    """
    For each pair (good_file, jb_file):
      - We load them both. Each is a dict with keys like 'hidden_states', 'attentions', etc.
      - We'll produce a single entry that has:
          'clean': { 'hidden_states': ..., 'attentions': ... },
          'typo':  { 'hidden_states': ..., 'attentions': ... }
        because the downstream build_dataset function expects .pt that looks like {clean:..., typo:...}.

    Return a list of these combined dicts.
    """
    combined_data = []
    for (g_path, j_path) in good_jb_pairs:
        try:
            good_dict = torch.load(g_path)
            jb_dict   = torch.load(j_path)

            # We'll store them in the shape the script wants:
            entry = {
                "clean": {
                    "hidden_states": good_dict["hidden_states"],
                    "attentions":    good_dict["attentions"],
                },
                "typo": {
                    "hidden_states": jb_dict["hidden_states"],
                    "attentions":    jb_dict["attentions"],
                }
            }
            # We could also copy topk_logits if we like, but the build_dataset logic doesn't use it.
            combined_data.append(entry)

        except Exception as e:
            logging.error(f"Error loading or merging {g_path} and {j_path}: {str(e)}")

    return combined_data


###############################################################################
# 4. Build Dataset (Same as Old Code, but uses the combined shape)
###############################################################################
def build_dataset(activation_data, extract_layers):
    """
    After loading clean_act / typo_act, we make sure to flatten any batch dimension
    so that each sample is always shape [hidden_dim].
    """
    layer_datasets = {l: [] for l in extract_layers}

    for entry in activation_data:
        clean_data = entry["clean"]
        jb_data    = entry["typo"]

        for l in extract_layers:
            layer_key = f"layer_{l}"

            clean_act = clean_data["hidden_states"].get(layer_key, None)
            jb_act    = jb_data["hidden_states"].get(layer_key, None)
            if clean_act is None or jb_act is None:
                continue

            clean_np = clean_act.to(torch.float32).cpu().numpy()
            jb_np    = jb_act.to(torch.float32).cpu().numpy()

            # handle each: if it’s 2D => (seq_len, hidden_dim), treat as batch=1
            # if it’s 3D => (batch_size, seq_len, hidden_dim)
            def process(arr, label):
                if arr.ndim == 2:
                    # shape (seq_len, hidden_dim) => single sample
                    arr = arr[np.newaxis, ...]  # => shape (1, seq_len, hidden_dim)
                elif arr.ndim != 3:
                    # skip any weird shape
                    logging.warning(f"Skipping: unexpected shape {arr.shape}")
                    return

                # now arr is (batch_size, seq_len, hidden_dim)
                batch_size = arr.shape[0]
                seq_len    = arr.shape[1]
                if seq_len == 0:
                    logging.warning(f"Skipping: zero tokens => {arr.shape}")
                    return

                for b_idx in range(batch_size):
                    vec = arr[b_idx].mean(axis=0)  # shape [hidden_dim]
                    if np.isnan(vec).any():
                        logging.warning("Skipping sample => got NaN in mean.")
                        continue
                    layer_datasets[l].append((vec, label))

            # process 'clean' as label=0, 'typo'/jb as label=1
            process(clean_np, 0)
            process(jb_np, 1)

    results = {}
    for l in extract_layers:
        data_list = layer_datasets[l]
        if not data_list:
            logging.warning(f"No valid data for layer {l}")
            continue

        X = np.array([item[0] for item in data_list], dtype=np.float32)
        y = np.array([item[1] for item in data_list], dtype=np.int64)
        results[l] = (X, y)

    return results

###############################################################################
# 5. Train/Evaluate 
###############################################################################
def train_and_evaluate_layer(X, y, layer_id, train_test_ratio=0.8):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_test_ratio, random_state=42, shuffle=True
    )
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, labels=[0, 1],
                                   target_names=["clean", "jb"], zero_division=0)

    msg = (
        f"Layer {layer_id} -> Acc: {acc:.4f}, F1: {f1:.4f}\n"
        f"{report}"
    )
    logging.info(msg)
    print(msg)

    return {
        "layer": layer_id,
        "accuracy": acc,
        "f1_score": f1,
        "report": report
    }


def main():
    # 1) Pair up the .pt files
    pairs = pair_activation_files(GOOD_DIR, JB_DIR)
    if not pairs:
        logging.error("No matched .pt files found across GOOD_DIR and JB_DIR.")
        return

    # 2) Load & combine
    combined = load_and_combine(pairs)
    if not combined:
        logging.error("No data loaded after combining. Exiting.")
        return

    # 3) Build dataset
    layer_data = build_dataset(combined, EXTRACT_HIDDEN_LAYERS)

    # 4) Train/evaluate for each layer
    all_results = []
    for layer_id, (X, y) in layer_data.items():
        res = train_and_evaluate_layer(X, y, layer_id, TRAIN_TEST_SPLIT_RATIO)
        all_results.append(res)

    # 5) Summarize
    summary = []
    for r in all_results:
        summary.append({
            "Layer": r["layer"],
            "Accuracy": r["accuracy"],
            "F1_score": r["f1_score"]
        })
    if summary:
        df = pd.DataFrame(summary)
        logging.info("Summary of results:\n" + df.to_string(index=False))
        print("\nSummary of results:")
        print(df.to_string(index=False))

    logging.info("Done. All results have been logged.")

if __name__ == "__main__":
    main()
