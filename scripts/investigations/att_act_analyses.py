import os
import torch
import logging
import glob
import json
import numpy as np
import csv
import math
from typing import Dict, Any, List
from collections import defaultdict

# ------------------------------------------------------------------------
# 1. Configuration and Setup
# ------------------------------------------------------------------------
PT_DATA_DIR = (
    os.environ.get("PT_DATA_DIR")
    or globals().get("PT_DATA_DIR")
    or "output/extractions/gemma9bit/attack"
)
# check: contains .pt files by original inference ! 
# (for the "attack" key). Each file something like:
# {
#   "hidden_states": { "layer_0": Tensor, "layer_5": Tensor, ... },
#   "attentions": { "layer_0": Tensor, "layer_5": Tensor, ... },
#   "topk_logits": Tensor,         # optional
#   "topk_indices": Tensor,        # optional
#   "input_ids": Tensor,           # shape [B, seq_len]
#   "final_predictions": [ ... ]   # list of strings
# }

ANALYSIS_OUTPUT_DIR = (
    os.environ.get("ANALYSIS_OUTPUT_DIR")
    or globals().get("ANALYSIS_OUTPUT_DIR")
    or "analyses/gemma9bit/attack"
)
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

HARMFUL_WORDS_FILE = (
    os.environ.get("HARMFUL_WORDS_FILE")
    or globals().get("HARMFUL_WORDS_FILE", "analyses/harmful_words.csv")
)

LOG_FILE = (
    os.environ.get("LOG_FILE")
    or globals().get("LOG_FILE")
    or "logs/analysis_attack.log"
)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# saved hidden states for layers [0, 5, 10, 15, 20, 25], - can list them here
EXTRACTED_LAYERS = [0, 5, 10, 15, 20, 25]

# ------------------------------------------------------------------------
# 2. LOGGING SETUP
# ------------------------------------------------------------------------
logger = logging.getLogger("AttackAnalysisLogger")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

logger.info("Starting **COMPREHENSIVE** 'attack' interpretability analysis...")

# ------------------------------------------------------------------------
# 3. LOAD HARMFUL WORDS
# ------------------------------------------------------------------------
harmful_words_set = set()
if HARMFUL_WORDS_FILE and os.path.isfile(HARMFUL_WORDS_FILE):
    logger.info(f"Loading harmful words from: {HARMFUL_WORDS_FILE}")
    with open(HARMFUL_WORDS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                harmful_words_set.add(word)
    logger.info(f"Loaded {len(harmful_words_set)} harmful words.")
else:
    logger.info("No harmful words file found or specified; continuing without it.")

# ------------------------------------------------------------------------
# 4. LOADING EXTRACTIONS
# ------------------------------------------------------------------------
def load_extraction_files(data_dir: str) -> List[Dict[str, Any]]:
    """
    Loads all .pt files from the specified directory, returning a list of 
    dictionaries with keys like 'hidden_states', 'attentions', 'input_ids', etc.
    """
    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    logger.info(f"Found {len(file_paths)} .pt files in {data_dir}.")
    if not file_paths:
        return []

    all_extractions = []
    for fp in file_paths:
        try:
            data = torch.load(fp, map_location="cpu")
            # data is a dict with:
            #   hidden_states (dict of layer -> Tensor)
            #   attentions (dict of layer -> Tensor)
            #   input_ids (Tensor)
            #   final_predictions (list of str)
            #   plus any others (topk_logits, topk_indices, etc.)
            all_extractions.append(data)
        except Exception as e:
            logger.exception(f"Error loading {fp}: {str(e)}")
    return all_extractions

# ------------------------------------------------------------------------
# 5. TOKENIZER LOADING
# ------------------------------------------------------------------------
# need the same model's tokenizer to decode input_ids. 
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # Adjust model name if needed to local path or HF model ID:
    MODEL_NAME = "google/gemma-2-9b-it"
    logger.info(f"Loading tokenizer for logit lens & token decoding: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # also load the model for the logit-lens approach:
    model_for_logit_lens = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    model_for_logit_lens.eval()
except Exception as e:
    logger.warning(f"Could not load the model/tokenizer from {MODEL_NAME}. Fallback. Error: {e}")
    class FallbackTokenizer:
        def convert_ids_to_tokens(self, ids):
            return [f"<tok_{i}>" for i in ids]
    tokenizer = FallbackTokenizer()
    model_for_logit_lens = None

# ------------------------------------------------------------------------
#                       HELPER: Decoding Token IDs
# ------------------------------------------------------------------------
def decode_tokens(token_ids: List[int]) -> List[str]:
    """
    Uses the loaded tokenizer to convert numeric IDs to tokens. 
    If we have a fallback, it'll produce <tok_#>.
    """
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return tokenizer.convert_ids_to_tokens(token_ids)
    else:
        return [f"<tok_{i}>" for i in token_ids]

# ------------------------------------------------------------------------
#                      1) TOKEN-LEVEL ATTENTION ANALYSIS
# ------------------------------------------------------------------------
def compute_token_level_attention(extractions):
    """
    For each sample in each extraction file, compute the average attention 
    distribution over tokens (summed across heads) for every (query, key) token pair.
    Returns a big list of rows (dict). We'll store them in CSV later.
    """
    results = []

    for file_idx, data in enumerate(extractions):
        input_ids = data["input_ids"]  # shape: [batch_size, seq_len]
        attentions = data["attentions"]  # dict: layer -> [bsz, n_heads, seq_len, seq_len]
        batch_size = input_ids.size(0)

        # For each layer
        for layer_key, attn_tensor in attentions.items():
            # shape: [batch_size, n_heads, seq_len, seq_len]
            attn_np = attn_tensor.float().numpy()
            bsz, n_heads, seq_len, _ = attn_np.shape

            # average over heads
            avg_attn = attn_np.mean(axis=1)  # shape [batch_size, seq_len, seq_len]

            for b in range(bsz):
                token_ids_b = input_ids[b].tolist()
                decoded_tokens_b = decode_tokens(token_ids_b)

                for query_idx in range(seq_len):
                    query_tok = decoded_tokens_b[query_idx]
                    for key_idx in range(seq_len):
                        key_tok = decoded_tokens_b[key_idx]
                        attn_val = float(avg_attn[b, query_idx, key_idx])
                        results.append({
                            "file_index": file_idx,
                            "sample_index": b,
                            "layer": layer_key,
                            "query_idx": query_idx,
                            "query_token": query_tok,
                            "key_idx": key_idx,
                            "key_token": key_tok,
                            "avg_attention": attn_val
                        })
    return results

# ------------------------------------------------------------------------
#       2) FRACTION OF ATTENTION ON HARMFUL WORDS (AGGREGATED LAYER-LEVEL)
# ------------------------------------------------------------------------
def compute_attention_on_harmful_words(extractions, harmful_words):
    """
    For each sample, for each layer, compute fraction of total attention 
    that goes to tokens matching harmful words (averaged over heads).
    """
    results = []

    # If no harmful words, weill fill fraction with None or 0
    if not harmful_words:
        for file_idx, data in enumerate(extractions):
            attentions = data["attentions"]
            input_ids = data["input_ids"]
            bsz = input_ids.size(0)
            for layer_key in attentions.keys():
                for b in range(bsz):
                    results.append({
                        "file_index": file_idx,
                        "sample_index": b,
                        "layer": layer_key,
                        "fraction_harmful": None
                    })
        return results

    harmful_words_lower = set(hw.lower() for hw in harmful_words)

    for file_idx, data in enumerate(extractions):
        input_ids = data["input_ids"]
        attentions = data["attentions"]
        bsz, seq_len = input_ids.size()

        # figure out which positions are harmful
        harmful_masks = []
        for b in range(bsz):
            tokens_b = decode_tokens(input_ids[b].tolist())
            mask_b = []
            for t in tokens_b:
                t_low = t.lower()
                # naive: if any harmful word is a substring
                is_harm = any(hw in t_low for hw in harmful_words_lower)
                mask_b.append(is_harm)
            harmful_masks.append(mask_b)

        # for each layer
        for layer_key, attn_tensor in attentions.items():
            attn_np = attn_tensor.float().numpy()  # shape [bsz, n_heads, seq_len, seq_len]
            _, n_heads, _, _ = attn_np.shape
            # average over heads
            avg_attn = attn_np.mean(axis=1)  # shape [bsz, seq_len, seq_len]

            for b in range(bsz):
                harm_positions = [i for i, x in enumerate(harmful_masks[b]) if x]
                if not harm_positions:
                    # no harmful tokens
                    results.append({
                        "file_index": file_idx,
                        "sample_index": b,
                        "layer": layer_key,
                        "fraction_harmful": 0.0
                    })
                    continue
                total_attn = 0.0
                harm_attn = 0.0
                for q_idx in range(seq_len):
                    row = avg_attn[b, q_idx]
                    row_sum = float(np.sum(row))
                    total_attn += row_sum
                    harm_sum = sum(row[pos] for pos in harm_positions)
                    harm_attn += harm_sum
                fraction = float(harm_attn / (total_attn + 1e-9))
                results.append({
                    "file_index": file_idx,
                    "sample_index": b,
                    "layer": layer_key,
                    "fraction_harmful": fraction
                })
    return results

# ------------------------------------------------------------------------
#      3) HEAD-SPECIFIC ANALYSIS: WHICH HEADS ATTEND STRONGLY TO HARMFUL?
# ------------------------------------------------------------------------
def compute_head_specific_attention_distribution(extractions, harmful_words):
    """
    For each layer, for each head, and for each sample, measure the fraction of 
    that head's total attention that goes to harmful tokens.
    """
    results = []
    harmful_words_lower = set(hw.lower() for hw in harmful_words)

    for file_idx, data in enumerate(extractions):
        input_ids = data["input_ids"]
        attentions = data["attentions"]
        bsz, seq_len = input_ids.size()

        # create harmful mask per sample
        masks = []
        for b in range(bsz):
            tokens_b = decode_tokens(input_ids[b].tolist())
            mask_b = [any(hw in t.lower() for hw in harmful_words_lower) for t in tokens_b]
            masks.append(mask_b)

        for layer_key, attn_tensor in attentions.items():
            attn_np = attn_tensor.float().numpy()  # [bsz, n_heads, seq_len, seq_len]
            bsz_, n_heads, _, _ = attn_np.shape
            for b in range(bsz_):
                harm_mask = masks[b]
                for h in range(n_heads):
                    matrix = attn_np[b, h]  # shape [seq_len, seq_len]
                    sum_all = float(np.sum(matrix))
                    if sum_all < 1e-12:
                        sum_all = 1e-12

                    if harmful_words:
                        sum_harm = 0.0
                        for q_idx in range(seq_len):
                            row = matrix[q_idx]
                            sum_harm += sum(row[k_idx] for k_idx, is_h in enumerate(harm_mask) if is_h)
                        fraction = float(sum_harm / sum_all)
                    else:
                        sum_harm = None
                        fraction = None
                    results.append({
                        "file_index": file_idx,
                        "sample_index": b,
                        "layer": layer_key,
                        "head": h,
                        "total_attention": float(sum_all),
                        "harmful_attention": float(sum_harm) if sum_harm is not None else None,
                        "fraction_harmful": fraction
                    })
    return results

# ------------------------------------------------------------------------
#   4) HIDDEN-STATE TRAJECTORY: HOW A TOKEN EVOLVES ACROSS LAYERS
# ------------------------------------------------------------------------
def compute_hidden_state_trajectories(extractions):
    """
    For each sample, for each token position, gather the hidden-state vectors 
    across extracted layers. That is, a "trajectory" from layer_0 -> layer_5
    -> layer_10, etc. We'll store them in a list of dicts.

    We'll write them to a CSV that just has metadata, 
    and store the actual vectors in a .npy for each sample if desired.
    """
    # We'll store metadata in a CSV, plus the actual vectors in separate .npy files.
    # The CSV will have columns:
    #   file_index, sample_index, token_index, token_text, layer -> path to vector .npy
    # or we can store the entire trajectory in one .npy as well.

    results = []
    # Will keep a structure: 
    #   hidden_trajectories[sample_global_id][token_index] = { layer: np.array(...) }

    sample_global_id = 0
    for file_idx, data in enumerate(extractions):
        input_ids = data["input_ids"]  # shape [bsz, seq_len]
        hidden_states_dict = data["hidden_states"]  # dict: "layer_0": [bsz, seq_len, hidden_dim], etc.

        bsz, seq_len = input_ids.size()

        # convert all hidden_states to numpy in a dictionary: layer_name -> np array
        # shape for each layer: (bsz, seq_len, hidden_dim)
        hidden_np_dict = {}
        for layer_key, tensor in hidden_states_dict.items():
            hidden_np_dict[layer_key] = tensor.float().numpy()

        for b in range(bsz):
            token_ids_b = input_ids[b].tolist()
            tokens_b = decode_tokens(token_ids_b)
            seq_length = len(token_ids_b)

            for t_idx in range(seq_length):
                t_tok = tokens_b[t_idx]
                # gather the trajectory across layers
                # e.g. layer_0 -> hidden_np_dict["layer_0"][b, t_idx, :]
                traj = {}
                for layer_key in sorted(hidden_np_dict.keys()):
                    vec = hidden_np_dict[layer_key][b, t_idx]  # shape [hidden_dim]
                    traj[layer_key] = vec

                # Will store a reference
                results.append({
                    "file_index": file_idx,
                    "sample_index": b,
                    "global_sample_id": sample_global_id,
                    "token_index": t_idx,
                    "token_text": t_tok,
                    "trajectory": traj  # can store the raw data here in Python
                })
            sample_global_id += 1

    return results

# ------------------------------------------------------------------------
#   5) COMPARING THE SAME TOKEN ACROSS MULTIPLE SAMPLES (HIDDEN STATES)
# ------------------------------------------------------------------------
def compare_same_token_across_samples(hidden_trajectory_results):
    """
    We gather hidden-state vectors for any repeated token (like 'hack', 'phish', etc.) 
    across the entire dataset, to measure similarity.

    We'll produce:
      - A CSV that for each token, we list sample_index, layer, vector norm, etc.
      - Optionally, we compute average vector or similarity metrics (like average 
        pairwise cosine similarity).
    """
    # Let's group by token_text
    token_groups = defaultdict(list)
    for row in hidden_trajectory_results:
        token_text = row["token_text"].lower()  # or keep original case
        token_groups[token_text].append(row)

    # Now for each token_text, have a list of sample references. 
    # Will do a naive approach:
    #  For each layer in the trajectory, gather vectors from all samples, compute:
    #   - average vector
    #   - store them so we can measure pairwise similarity
    # Will store final results in a CSV, plus we can store the average vector in .npy

    aggregated_results = []
    for token_text, instances in token_groups.items():
        # if a token only appears once, there's not much to compare
        if len(instances) < 2:
            continue

        # We know each row has row["trajectory"] which is {layer_key: np.array(...) }
        # Will gather them by layer.
        # e.g. layer -> list of np arrays
        layer_vectors = defaultdict(list)
        for inst in instances:
            for layer_key, vec in inst["trajectory"].items():
                layer_vectors[layer_key].append(vec)

        # compute average vectors and maybe pairwise similarities
        # Will store them in aggregated_results
        for layer_key, vecs in layer_vectors.items():
            stack = np.stack(vecs, axis=0)  # shape [num_occurrences, hidden_dim]
            mean_vec = np.mean(stack, axis=0)
            # measure average pairwise cos similarity
            # cos sim for x,y is x.dot(y)/(||x||*||y||)
            norms = np.linalg.norm(stack, axis=1, keepdims=True)
            normed = stack / (norms + 1e-9)
            sims = np.matmul(normed, normed.T)  # shape [n, n]
            # only want upper triangle mean (excluding diagonal)
            n = sims.shape[0]
            sim_sum = 0.0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    sim_sum += sims[i, j]
                    count += 1
            avg_pairwise_sim = sim_sum / (count + 1e-9)

            aggregated_results.append({
                "token_text": token_text,
                "layer_key": layer_key,
                "count_occurrences": len(vecs),
                "avg_pairwise_cos_sim": float(avg_pairwise_sim)
            })

    return aggregated_results

# ------------------------------------------------------------------------
#    6) DIMENSIONALITY REDUCTION (PCA) ON REPEATED TOKEN VECTORS
# ------------------------------------------------------------------------
def run_pca_on_repeated_tokens(hidden_trajectory_results, target_token: str = "hack"):
    """
    Example function: for a chosen token (like 'hack'), gather all hidden-state vectors 
    across all samples and layers, then run PCA to reduce to 2D, to see if there's 
    a pattern across layers/samples. We'll store the 2D coords in a CSV.
    """
    # Will gather a big list of vectors, each with fields:
    #   - file_index, sample_index, layer_key, vector (ndarray)
    big_list = []
    for row in hidden_trajectory_results:
        tok_low = row["token_text"].lower()
        if tok_low == target_token.lower():
            for layer_key, vec in row["trajectory"].items():
                big_list.append({
                    "file_index": row["file_index"],
                    "sample_index": row["sample_index"],
                    "token_index": row["token_index"],
                    "layer_key": layer_key,
                    "vector": vec
                })

    if not big_list:
        logger.info(f"No occurrences of token '{target_token}' found. Skipping PCA.")
        return []

    # collect all vectors into X
    X = np.stack([d["vector"] for d in big_list], axis=0)  # shape [N, hidden_dim]
    # run PCA (we can do 2D) - for now quick SVD-based PCA.
    X_mean = X.mean(axis=0, keepdims=True)
    X_centered = X - X_mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # PC dims
    pc = np.matmul(X_centered, Vt[:2].T)  # shape [N, 2]

    # Attach 2D coords back to big_list
    results = []
    for i, rowdat in enumerate(big_list):
        x_ = float(pc[i, 0])
        y_ = float(pc[i, 1])
        results.append({
            "file_index": rowdat["file_index"],
            "sample_index": rowdat["sample_index"],
            "token_index": rowdat["token_index"],
            "layer_key": rowdat["layer_key"],
            "pc1": x_,
            "pc2": y_
        })

    return results

# ------------------------------------------------------------------------
#   7) LOGIT LENS: APPLY MODEL HEAD TO INTERMEDIATE HIDDEN STATES
# ------------------------------------------------------------------------
def apply_logit_lens(extractions, model):
    """
    For each sample, for each layer, take the hidden state at that layer 
    (for each token) and apply the model's final linear layer (the unembedding) 
    to see which tokens are likely next. We'll do a top-5 decode as an example.

    We'll produce rows:
        file_index, sample_index, layer, token_index, token_text, top_predictions
    where top_predictions is something like "['the', 'a', 'of', ...]"

    This requires we have 'model_for_logit_lens' loaded (the same model).
    """
    if model is None:
        logger.warning("No model loaded for logit lens. Skipping.")
        return []

    # Will access the unembedding matrix usually via model.lm_head.weight
    # or model.transformer.wte, etc., depending on architecture.
    # For standard GPT-like models in huggingface: final_lin = model.lm_head
    final_lin = model.lm_head  # linear layer: hidden_dim -> vocab_size
    vocab_size = final_lin.weight.size(0)
    hidden_dim = final_lin.weight.size(1)

    results = []

    for file_idx, data in enumerate(extractions):
        input_ids = data["input_ids"]
        hidden_dict = data["hidden_states"]  # layer -> [bsz, seq_len, hidden_dim]
        bsz, seq_len = input_ids.size()

        for b in range(bsz):
            token_ids_b = input_ids[b].tolist()
            tokens_b = decode_tokens(token_ids_b)

            for layer_key, h_tensor in hidden_dict.items():
                # shape: [bsz, seq_len, hidden_dim]
                # extract sample b
                hidden_np = h_tensor[b].float().numpy()  # shape [seq_len, hidden_dim]

                for t_idx in range(seq_len):
                    h_vec = hidden_np[t_idx]  # shape [hidden_dim]
                    # Will do a manual forward: logits = final_lin(h_vec)
                    # in python: logits = h_vec @ final_lin.weight.T + final_lin.bias
                    # but we can do it with torch as well:
                    with torch.no_grad():
                        h_t = torch.from_numpy(h_vec).float().unsqueeze(0)  # shape [1, hidden_dim]
                        # shape [1, vocab_size]
                        logits_t = torch.matmul(h_t, final_lin.weight.transpose(0, 1))
                        if final_lin.bias is not None:
                            logits_t = logits_t + final_lin.bias

                    # get top-5
                    top_vals, top_ids = torch.topk(logits_t, k=5, dim=-1)
                    top_ids_list = top_ids[0].tolist()
                    # decode them
                    top_tokens = tokenizer.convert_ids_to_tokens(top_ids_list) if hasattr(tokenizer, "convert_ids_to_tokens") else [str(i) for i in top_ids_list]
                    results.append({
                        "file_index": file_idx,
                        "sample_index": b,
                        "layer": layer_key,
                        "token_index": t_idx,
                        "token_text": tokens_b[t_idx],
                        "top_tokens": top_tokens
                    })
    return results

# ------------------------------------------------------------------------
#                        MASTER FUNCTION: RUN EVERYTHING
# ------------------------------------------------------------------------
def run_comprehensive_analysis():
    logger.info(f"Loading extracted activations from: {PT_DATA_DIR}")
    extractions = load_extraction_files(PT_DATA_DIR)

    if not extractions:
        logger.error("No extraction data loaded. Aborting analysis.")
        return

    #########################################
    # 1) TOKEN-LEVEL ATTENTION
    #########################################
    logger.info("** 1) TOKEN-LEVEL ATTENTION ANALYSIS **")
    token_attn_rows = compute_token_level_attention(extractions)
    token_attn_csv = os.path.join(ANALYSIS_OUTPUT_DIR, "token_level_attention.csv")
    with open(token_attn_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file_index", "sample_index", "layer", 
            "query_idx", "query_token",
            "key_idx", "key_token",
            "avg_attention"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in token_attn_rows:
            writer.writerow(row)
    logger.info(f"[SAVED] Token-level attention -> {token_attn_csv}")

    #########################################
    # 2) ATTENTION ON HARMFUL WORDS
    #########################################
    logger.info("** 2) ATTENTION ON HARMFUL WORDS **")
    frac_harmful_rows = compute_attention_on_harmful_words(extractions, harmful_words_set)
    frac_harmful_csv = os.path.join(ANALYSIS_OUTPUT_DIR, "fraction_harmful_attention.csv")
    with open(frac_harmful_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["file_index", "sample_index", "layer", "fraction_harmful"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in frac_harmful_rows:
            writer.writerow(row)
    logger.info(f"[SAVED] Fraction-of-harmful-attention -> {frac_harmful_csv}")

    #########################################
    # 3) HEAD-SPECIFIC ANALYSIS
    #########################################
    logger.info("** 3) HEAD-SPECIFIC ATTENTION ANALYSIS **")
    head_dist_rows = compute_head_specific_attention_distribution(extractions, harmful_words_set)
    head_dist_csv = os.path.join(ANALYSIS_OUTPUT_DIR, "head_specific_attention.csv")
    with open(head_dist_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file_index", "sample_index", "layer", "head",
            "total_attention", "harmful_attention", "fraction_harmful"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in head_dist_rows:
            writer.writerow(row)
    logger.info(f"[SAVED] Head-specific analysis -> {head_dist_csv}")

    #########################################
    # 4) HIDDEN-STATE TRAJECTORIES
    #########################################
    logger.info("** 4) HIDDEN-STATE TRAJECTORY COLLECTION **")
    hidden_traj = compute_hidden_state_trajectories(extractions)
    # Will store a CSV with "file_index, sample_index, global_sample_id, token_index, token_text"
    # The actual layer->vector mapping is large, so let's store it in a .pt or .npy.
    # Will store row['trajectory'] in a separate .npy file for each row to avoid huge CSVs.
    traj_csv = os.path.join(ANALYSIS_OUTPUT_DIR, "hidden_state_trajectories.csv")
    with open(traj_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file_index", "sample_index", "global_sample_id",
            "token_index", "token_text", "npy_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(hidden_traj):
            out_npy_path = os.path.join(
                ANALYSIS_OUTPUT_DIR, 
                f"trajectory_{row['file_index']}_{row['sample_index']}_{row['token_index']}.npy"
            )
            np.save(out_npy_path, row["trajectory"])  # saves a dict of layer->vector as an object
            writer.writerow({
                "file_index": row["file_index"],
                "sample_index": row["sample_index"],
                "global_sample_id": row["global_sample_id"],
                "token_index": row["token_index"],
                "token_text": row["token_text"],
                "npy_path": out_npy_path
            })
    logger.info(f"[SAVED] Hidden-state trajectories -> {traj_csv} + individual .npy files")

    #########################################
    # 5) COMPARE THE SAME TOKEN ACROSS SAMPLES
    #########################################
    logger.info("** 5) COMPARING REPEATED TOKENS (HIDDEN STATES) **")
    # 'hidden_traj' has all the data we need.
    # Will do a single function that aggregates repeated tokens.
    aggregated_repeats = compare_same_token_across_samples(hidden_traj)
    repeats_csv = os.path.join(ANALYSIS_OUTPUT_DIR, "repeated_token_similarity.csv")
    with open(repeats_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["token_text", "layer_key", "count_occurrences", "avg_pairwise_cos_sim"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_repeats:
            writer.writerow(row)
    logger.info(f"[SAVED] Repeated-token similarity -> {repeats_csv}")

    #########################################
    # 6) PCA ON A CHOSEN TOKEN (E.G. "hack")
    #########################################
    # can choose a token suspected relevant. - now, "hack".
    target_token_for_pca = "hack"
    logger.info(f"** 6) PCA on repeated token '{target_token_for_pca}' **")
    pca_coords = run_pca_on_repeated_tokens(hidden_traj, target_token=target_token_for_pca)
    if pca_coords:
        pca_csv = os.path.join(ANALYSIS_OUTPUT_DIR, f"pca_{target_token_for_pca}.csv")
        with open(pca_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "file_index", "sample_index", "token_index", "layer_key", "pc1", "pc2"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in pca_coords:
                writer.writerow(row)
        logger.info(f"[SAVED] PCA coords for token='{target_token_for_pca}' -> {pca_csv}")
    else:
        logger.info(f"No occurrences of token '{target_token_for_pca}' found, skipping PCA saving.")

    #########################################
    # 7) LOGIT LENS
    #########################################
    logger.info("** 7) LOGIT LENS: Checking top-5 next tokens from each layer's hidden state **")
    logit_lens_rows = apply_logit_lens(extractions, model_for_logit_lens)
    if logit_lens_rows:
        logit_lens_csv = os.path.join(ANALYSIS_OUTPUT_DIR, "logit_lens.csv")
        with open(logit_lens_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["file_index", "sample_index", "layer", "token_index", "token_text", "top_tokens"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in logit_lens_rows:
                writer.writerow(row)
        logger.info(f"[SAVED] Logit-lens analysis -> {logit_lens_csv}")
    else:
        logger.info("No logit-lens results (likely no model available).")

    #########################################
    # 8) MULTI-FILE AGGREGATED COMPARISONS
    #########################################
    logger.info("** 8) MULTI-FILE AGGREGATED COMPARISON **")
    # can do some summary stats across all the loaded .pt files:
    # for now: average fraction of harmful attention per layer, across the entire dataset
    fraction_by_layer = defaultdict(list)  # layer -> list of fractions
    for row in frac_harmful_rows:
        fraction_by_layer[row["layer"]].append(row["fraction_harmful"])

    multi_file_csv = os.path.join(ANALYSIS_OUTPUT_DIR, "multi_file_aggregates.csv")
    with open(multi_file_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["layer", "mean_fraction_harmful", "std_fraction_harmful", "count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for layer, vals in fraction_by_layer.items():
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            count = len(vals)
            writer.writerow({
                "layer": layer,
                "mean_fraction_harmful": mean_val,
                "std_fraction_harmful": std_val,
                "count": count
            })
    logger.info(f"[SAVED] Multi-file aggregated stats -> {multi_file_csv}")

    logger.info("All advanced analyses completed. Check output CSVs & .npy files.")

# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------
if __name__ == "__main__":
    run_comprehensive_analysis()
