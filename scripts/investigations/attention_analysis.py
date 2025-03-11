#!/usr/bin/env python3

import os
import re
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
MODEL_NAME = (
    os.environ.get("MODEL_NAME")
    or globals().get("MODEL_NAME")
    or "google/gemma-2-9b-it"
)
DATA_FIRST_DIR = (
    os.environ.get("DATA_FIRST_DIR")
    or globals().get("DATA_FIRST_DIR")
    or "output/extractions/gemma2bit/attack"
)
DATA_SECOND_DIR = (
    os.environ.get("DATA_SECOND_DIR")
    or globals().get("DATA_SECOND_DIR")
    or "output/extractions/gemma2bit/jailbreak"
)
OUTPUT_DIR = (
    os.environ.get("OUTPUT_DIR")
    or globals().get("OUTPUT_DIR")
    or "output/attention/attack_jailbreak"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# “Nice” words to look for (lowercased)
BAD_WORDS_CSV = "analyses/bad_words.csv"

# Load bad words from CSV (for one word/phrase per line, no header):
with open(BAD_WORDS_CSV, "r", encoding="utf-8") as f:
    BAD_WORDS = [line.strip() for line in f if line.strip()]

# Instead pandas:
# df_bad = pd.read_csv(BAD_WORDS_CSV, header=None)
# BAD_WORDS = df_bad[0].dropna().astype(str).str.strip().tolist()

# If you want to use robust subword merging for bad-word detection, set this True
ROBUST_BAD_DETECTION = True

# Taking same tokenizer that was used originally to decode input_ids. 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------
def get_pt_files(directory):
    """Return sorted list of .pt files in a directory."""
    files = glob.glob(os.path.join(directory, "activations_*.pt"))
    files = sorted(files)  # sorts by name, e.g. activations_00000_00004.pt
    return files

def load_activations(pt_file):
    """
    Loads a single .pt file. 
    We expect structure like:
      {
         "attentions": { "layer_0": <tensor>, "layer_5": <tensor>, ...},
         "hidden_states": ...,
         "input_ids": <tensor>,
         "topk_logits": ...,
         "topk_indices": ...,
         "final_predictions": [...]
      }
    """
    return torch.load(pt_file)


# ====================== BAD-WORD DETECTION ======================

def find_bad_tokens_subword(decoded_tokens, bad_words):
    """
    Attempt to handle subword splitting so that 
    'please' => ['pl', 'ea', 'se'] is recognized as 'please'.

    Approach:
      - We accumulate subwords into a 'word_buffer' until we hit punctuation 
        or a boundary (non-alphanumeric).
      - Then compare the merged buffer to each BAD_WORDS item (after cleaning punctuation).
      - If it matches, mark all subtokens in that word as "bad."

    This won't be perfect for all tokenizers, but it helps more than naive substring checks.
    """
    bad_mask = [False] * len(decoded_tokens)

    word_buffer = []
    word_start_idx = 0

    def flush_word(buffer, start_idx, end_idx):
        # Merge subword pieces (remove typical filler like "▁", "Ġ" if they appear).
        merged = "".join(buffer).replace("▁", "").replace("Ġ", "").lower()
        # Remove punctuation to compare more cleanly
        merged_clean = re.sub(r"\W+", "", merged)

        for w in bad_words:
            w_clean = re.sub(r"\W+", "", w.lower())
            if merged_clean == w_clean:
                # Mark all tokens that contributed to this word
                for idx in range(start_idx, end_idx):
                    bad_mask[idx] = True

    # We'll define a helper that decides if a token is "alphanumeric enough"
    def is_wordish(token: str):
        # This is just a simple check. If it's purely punctuation or special, return False
        # If it has letters/numbers, consider it part of a "word" buffer.
        cleaned = re.sub(r"\W+", "", token)  # remove non-alphanumeric
        return len(cleaned) > 0

    for i, tok in enumerate(decoded_tokens):
        if is_wordish(tok):
            word_buffer.append(tok)
        else:
            # We hit punctuation or a boundary => flush the buffer as a "word"
            if word_buffer:
                flush_word(word_buffer, word_start_idx, i)
                word_buffer = []
            word_start_idx = i + 1

    # End of sequence flush
    if word_buffer:
        flush_word(word_buffer, word_start_idx, len(decoded_tokens))

    return bad_mask


# ====================== IDENTIFY TOKEN RANGES (TWO VERSIONS) ======================

def identify_token_ranges_naive(token_ids):
    """
    Original naive approach:
      - We label any token that *contains* one of the BAD_WORDS as "bad".
      - We label tokens after the first colon as "math".
    """
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    decoded_tokens_lower = [t.lower() for t in decoded_tokens]

    # Where is the first colon?
    colon_positions = [i for i, tok in enumerate(decoded_tokens) if ":" in tok]
    first_colon_pos = colon_positions[0] if len(colon_positions) > 0 else None

    # Mark bad tokens (naive substring check)
    bad_mask = [False] * len(decoded_tokens)
    for i, tok_l in enumerate(decoded_tokens_lower):
        for w in BAD_WORDS:
            if w in tok_l:  # substring
                bad_mask[i] = True
                break

    # Mark math tokens
    math_mask = [False] * len(decoded_tokens)
    if first_colon_pos is not None:
        for i in range(first_colon_pos+1, len(decoded_tokens)):
            math_mask[i] = True

    return {
        "decoded_tokens": decoded_tokens,
        "bad_mask": bad_mask,
        "math_mask": math_mask,
        "colon_pos": first_colon_pos
    }

def identify_token_ranges_robust(token_ids):
    """
    Uses a more robust subword merging for "bad" words.
    Otherwise, colon detection is same as naive approach.
    """
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)

    # Identify colon positions
    colon_positions = [i for i, tok in enumerate(decoded_tokens) if ":" in tok]
    first_colon_pos = colon_positions[0] if len(colon_positions) > 0 else None

    # More robust bad-word detection
    bad_mask = find_bad_tokens_subword(decoded_tokens, BAD_WORDS)

    # Mark math tokens
    math_mask = [False] * len(decoded_tokens)
    if first_colon_pos is not None:
        for i in range(first_colon_pos+1, len(decoded_tokens)):
            math_mask[i] = True

    return {
        "decoded_tokens": decoded_tokens,
        "bad_mask": bad_mask,
        "math_mask": math_mask,
        "colon_pos": first_colon_pos
    }


# ====================== ATTENTION EXTRACTION (AGGREGATE) ======================

def extract_attention_stats(attentions, input_ids):
    """
    Returns:
      {
        "per_layer": {layer_0: [B, seq_len], layer_1: [B, seq_len], ...}
        "avg_layers": [B, seq_len] (averaged across all extracted layers)
        "input_ids": input_ids
      }
    Where each [B, seq_len] is the average attention each token i receives from all heads, 
    summing over source tokens, then averaging over heads.
    """
    stats_per_layer = {}

    for layer_name, attn_tensor in attentions.items():
        # shape: [B, H, seq_len, seq_len]
        # sum over seq_len dimension=2 => shape [B, H, seq_len]
        # then mean over heads dimension=1 => shape [B, seq_len]
        attn_sum = attn_tensor.sum(dim=2)
        attn_mean = attn_sum.mean(dim=1)
        stats_per_layer[layer_name] = attn_mean  # [B, seq_len]

    # average over layers
    layer_list = list(stats_per_layer.values())  # each is [B, seq_len]
    if len(layer_list) == 0:
        raise ValueError("No layers found in `attentions` dictionary.")
    all_layers = torch.stack(layer_list, dim=0)  # shape [num_layers, B, seq_len]
    avg_across_layers = all_layers.mean(dim=0)   # [B, seq_len]

    return {
        "per_layer": stats_per_layer,
        "avg_layers": avg_across_layers,
        "input_ids": input_ids
    }


# ====================== ATTENTION EXTRACTION (PER-LAYER, PER-HEAD) ======================

def extract_per_layer_head_stats(attentions, input_ids):
    """
    Returns a list of dicts, each containing:
      {
        'layer': L,
        'head': H,
        'batch_idx': b,
        'attn_sum': [seq_len]  # sum of attention over the "source" dimension
        'attn_mean': [seq_len] # or other metric if you want
      }
    so you can compute your own "bad token" or "math token" average later.

    We'll define "attn_sum[i]" as how much attention token i pays to all tokens j 
    (summing across j). That is attn[b, h, i, :].sum() in typical transformers notation.

    Then if you want "attention each token receives," you might sum across i. 
    But let's keep the logic consistent with your existing approach: 
    we sum dimension=3 => shape [B, H, seq_len].
    """
    results = []
    # Sort layer keys so we iterate in ascending layer order
    sorted_layers = sorted(attentions.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    B, S = input_ids.shape

    for layer_name in sorted_layers:
        attn_tensor = attentions[layer_name]  # shape [B, H, seq_len, seq_len]
        b_size, h_size, seq_len, seq_len2 = attn_tensor.shape
        # sanity check
        if b_size != B or seq_len != S or seq_len2 != S:
            raise ValueError(f"Shape mismatch in {layer_name}. Expected [B,H,{S},{S}] got {attn_tensor.shape}")

        # parse out integer layer_idx from "layer_0"
        layer_idx = int(layer_name.split('_')[1]) if '_' in layer_name else layer_name

        # sum across the "source" tokens => shape [B, H, seq_len]
        # so attn_sum[b, h, i] = sum_j attn[b, h, i, j]
        attn_sum = attn_tensor.sum(dim=3)  # [B, H, seq_len]

        # we can also compute mean across j if we want
        # attn_mean = attn_tensor.mean(dim=3)  # [B, H, seq_len]
        # but let's store both
        attn_mean = attn_tensor.mean(dim=3)

        # collect
        for b_idx in range(B):
            for h_idx in range(h_size):
                row = {
                    "layer": layer_idx,
                    "head": h_idx,
                    "batch_idx": b_idx,
                    # We'll store these as lists for now. 
                    # We could also store them as numpy arrays.
                    "attn_sum": attn_sum[b_idx, h_idx].cpu().tolist(),
                    "attn_mean": attn_mean[b_idx, h_idx].cpu().tolist()
                }
                results.append(row)

    return results


# ====================== MAIN SCRIPT ======================

def main():
    # 1) Gather the file paths
    attack_files = get_pt_files(DATA_FIRST_DIR)
    jailbreak_files = get_pt_files(DATA_SECOND_DIR)

    print(f"Found {len(attack_files)} attack .pt files.")
    print(f"Found {len(jailbreak_files)} jailbreak .pt files.")

    # These will accumulate stats for the "aggregate" approach
    attention_records_attack = []
    attention_records_jailbreak = []

    # Also gather per-layer-head detailed stats
    plh_records_attack = []  # "Per-Layer-Head" records
    plh_records_jailbreak = []

    # We'll define a helper function to unify repeated logic
    def process_files(pt_files, prompt_type):
        """
        Returns two lists of dictionaries:
          1) The aggregate summary records
          2) The per-layer-head records
        """
        agg_records = []
        layer_head_records = []

        for pt_file in pt_files:
            data = load_activations(pt_file)
            # data["attentions"]: dict {layer_0: [B, H, S, S], ...}
            # data["input_ids"]: [B, S]

            # 1) AGGREGATE STATS
            batch_attn_stats = extract_attention_stats(data["attentions"], data["input_ids"])
            B, S = batch_attn_stats["avg_layers"].shape

            # Decide which identify function we use for "bad" detection
            # We'll do robust if ROBUST_BAD_DETECTION = True, else naive
            identify_fn = identify_token_ranges_robust if ROBUST_BAD_DETECTION else identify_token_ranges_naive

            for i in range(B):
                #attn_on_tokens = batch_attn_stats["avg_layers"][i].numpy()  # shape [seq_len]
                attn_on_tokens = batch_attn_stats["avg_layers"][i].float().cpu().numpy()

                input_ids_row = batch_attn_stats["input_ids"][i].tolist()

                masks_info = identify_fn(input_ids_row)
                bad_mask = np.array(masks_info["bad_mask"], dtype=bool)
                math_mask = np.array(masks_info["math_mask"], dtype=bool)

                avg_attn_bad = attn_on_tokens[bad_mask].mean() if bad_mask.any() else 0.0
                avg_attn_math = attn_on_tokens[math_mask].mean() if math_mask.any() else 0.0
                avg_attn_all = attn_on_tokens.mean()

                agg_records.append({
                    "batch_file": pt_file,
                    "avg_attn_bad": avg_attn_bad,
                    "avg_attn_math": avg_attn_math,
                    "avg_attn_all": avg_attn_all,
                    "num_bad_tokens": bad_mask.sum(),
                    "num_math_tokens": math_mask.sum(),
                    "prompt_type": prompt_type
                })

            # 2) PER-LAYER-HEAD STATS
            plh_data = extract_per_layer_head_stats(data["attentions"], data["input_ids"])
            # plh_data is a list of dicts, each has "layer", "head", "batch_idx", plus "attn_sum", "attn_mean" as lists
            for row in plh_data:
                # We'll figure out the "bad_mask" and "math_mask" for this sample
                b_idx = row["batch_idx"]
                input_ids_row = data["input_ids"][b_idx].tolist()
                masks_info = identify_fn(input_ids_row)
                bad_mask = np.array(masks_info["bad_mask"], dtype=bool)
                math_mask = np.array(masks_info["math_mask"], dtype=bool)

                # row["attn_sum"] is a python list of length seq_len
                attn_sum_array = np.array(row["attn_sum"], dtype=np.float32)  # shape [seq_len]
                attn_mean_array = np.array(row["attn_mean"], dtype=np.float32)

                # For now, doing "avg attention on bad tokens" for sum and mean
                # We can do many variants if you want, e.g. "attention *maybe* all? 
                # But now keeping it consistent with approach: sum across j => how much token i attends 
                # to the entire sequence. Then we average those i's that are "bad."
                # If you actually want "which tokens are receiving attention," you'd do a different dimension.
                # We'll store it anyway:

                avg_sum_bad = attn_sum_array[bad_mask].mean() if bad_mask.any() else 0.0
                avg_sum_math = attn_sum_array[math_mask].mean() if math_mask.any() else 0.0
                avg_sum_all  = attn_sum_array.mean()

                avg_mean_bad = attn_mean_array[bad_mask].mean() if bad_mask.any() else 0.0
                avg_mean_math = attn_mean_array[math_mask].mean() if math_mask.any() else 0.0
                avg_mean_all  = attn_mean_array.mean()

                new_row = {
                    "layer": row["layer"],
                    "head": row["head"],
                    "batch_idx": b_idx,
                    "prompt_type": prompt_type,
                    "avg_sum_bad": float(avg_sum_bad),
                    "avg_sum_math": float(avg_sum_math),
                    "avg_sum_all": float(avg_sum_all),
                    "avg_mean_bad": float(avg_mean_bad),
                    "avg_mean_math": float(avg_mean_math),
                    "avg_mean_all": float(avg_mean_all),
                    "batch_file": pt_file
                }
                layer_head_records.append(new_row)

        return agg_records, layer_head_records

    # Process attack files
    attack_agg, attack_plh = process_files(attack_files, prompt_type="attack")
    attention_records_attack.extend(attack_agg)
    plh_records_attack.extend(attack_plh)

    # Process jailbreak files
    jailbreak_agg, jailbreak_plh = process_files(jailbreak_files, prompt_type="jailbreak")
    attention_records_jailbreak.extend(jailbreak_agg)
    plh_records_jailbreak.extend(jailbreak_plh)

    # 4) Create DataFrames for AGGREGATE
    df_attack = pd.DataFrame(attention_records_attack)
    df_jailbreak = pd.DataFrame(attention_records_jailbreak)
    df_all = pd.concat([df_attack, df_jailbreak], ignore_index=True)

    # 5) AGGREGATE Stats
    mean_stats = df_all.groupby("prompt_type")[["avg_attn_bad","avg_attn_math","avg_attn_all"]].mean()
    print("=== Overall Mean Attention Stats (Aggregate) ===")
    print(mean_stats)

    # Save stats to CSV
    stats_csv_path = os.path.join(OUTPUT_DIR, "attention_summary_aggregate.csv")
    mean_stats.to_csv(stats_csv_path)
    print(f"Saved AGGREGATE summary CSV to: {stats_csv_path}")

    # 6) AGGREGATE Plots

    # a) Compare average attention on bad words
    plt.figure()
    df_all.boxplot(column="avg_attn_bad", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Aggregate - Average Attention on Nice Words")
    plt.ylabel("Attention (mean over sequence for bad tokens)")
    plt.xlabel("Prompt Type")
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_aggregate_attn_on_bad_words.png"))
    plt.close()

    # b) Compare average attention on math portion
    plt.figure()
    df_all.boxplot(column="avg_attn_math", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Aggregate - Average Attention on Math Portion")
    plt.ylabel("Attention (mean for tokens after colon)")
    plt.xlabel("Prompt Type")
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_aggregate_attn_on_math.png"))
    plt.close()

    # c) Compare overall average attention
    plt.figure()
    df_all.boxplot(column="avg_attn_all", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Aggregate - Overall Average Attention on Prompt")
    plt.ylabel("Attention (mean over all tokens)")
    plt.xlabel("Prompt Type")
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_aggregate_attn_on_all.png"))
    plt.close()

    # d) Histograms for distribution of "avg_attn_bad"
    for prompt_type in df_all["prompt_type"].unique():
        subset = df_all[df_all["prompt_type"] == prompt_type]["avg_attn_bad"]
        plt.figure()
        subset.hist(bins=20)
        plt.title(f"Distribution of Avg Attn on Nice - {prompt_type}")
        plt.xlabel("Attention Score")
        plt.ylabel("Count")
        plt.savefig(os.path.join(OUTPUT_DIR, f"hist_avg_attn_bad_{prompt_type}.png"))
        plt.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                 PER-LAYER-HEAD ANALYSIS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df_plh_attack = pd.DataFrame(plh_records_attack)
    df_plh_jailbreak  = pd.DataFrame(plh_records_jailbreak)
    df_plh_all    = pd.concat([df_plh_attack, df_plh_jailbreak], ignore_index=True)

    # For a quick summary, let's compute the mean "avg_sum_bad" by layer, by prompt_type
    plh_mean_stats = df_plh_all.groupby(["prompt_type","layer"])[["avg_sum_bad","avg_sum_math","avg_sum_all"]].mean()
    print("=== Per-Layer-Head Stats (Mean of 'attn_sum' metrics) ===")
    print(plh_mean_stats)

    # Save to CSV
    plh_csv_path = os.path.join(OUTPUT_DIR, "attention_summary_per_layer_head.csv")
    plh_mean_stats.to_csv(plh_csv_path)
    print(f"Saved PER-LAYER-HEAD summary CSV to: {plh_csv_path}")

    # Let's produce a boxplot of "avg_sum_bad" by layer, for each prompt type
    # We'll do one figure per prompt type, with layer on the x-axis.
    for prompt_type in df_plh_all["prompt_type"].unique():
        data_sub = df_plh_all[df_plh_all["prompt_type"] == prompt_type]

        plt.figure()
        data_sub.boxplot(column="avg_sum_bad", by="layer", grid=False)
        plt.suptitle("")
        plt.title(f"Per-Layer - avg_sum_bad - {prompt_type}")
        plt.ylabel("avg_sum_bad (Attn over all tokens for bad indices)")
        plt.xlabel("Layer")
        plt.savefig(os.path.join(OUTPUT_DIR, f"box_perlayer_sum_bad_{prompt_type}.png"))
        plt.close()

    # We can also do a single figure per layer to see distribution across heads
    # That means for each layer, do boxplot column="avg_sum_bad", by="head".
    # We'll do it for each prompt type as well => many plots:
    unique_layers = sorted(df_plh_all["layer"].unique())
    for layer in unique_layers:
        for prompt_type in df_plh_all["prompt_type"].unique():
            subset = df_plh_all[(df_plh_all["layer"] == layer) & (df_plh_all["prompt_type"] == prompt_type)]
            if subset.empty:
                continue

            plt.figure()
            subset.boxplot(column="avg_sum_bad", by="head", grid=False)
            plt.suptitle("")
            plt.title(f"Layer {layer} - {prompt_type} - avg_sum_bad by head")
            plt.ylabel("avg_sum_bad (Sum over tokens, subset=bad_mask)")
            plt.xlabel("Head")
            plt.savefig(os.path.join(OUTPUT_DIR, f"box_layer{layer}_{prompt_type}_sum_bad_byhead.png"))
            plt.close()

    # Could do the same for avg_mean_bad, avg_sum_math, etc. 
    # Just replicate with different columns.

    print("All plots saved to:", OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
