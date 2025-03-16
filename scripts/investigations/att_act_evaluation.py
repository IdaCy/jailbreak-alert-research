#!/usr/bin/env python3
import os
import csv
import glob
import json
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------------------
# 1. Configuration and Setup
# ------------------------------------------------------------------------
ANALYSIS_OUTPUT_DIR = (
    os.environ.get("ANALYSIS_OUTPUT_DIR")
    or globals().get("ANALYSIS_OUTPUT_DIR")
    or "analyses/gemma9bit/attack"
)
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

RESULTS_SUMMARY_DIR = (
    os.environ.get("RESULTS_SUMMARY_DIR")
    or globals().get("RESULTS_SUMMARY_DIR")
    or "analyses/gemma9bit/summary_attack"
)
os.makedirs(RESULTS_SUMMARY_DIR, exist_ok=True)

LOG_FILE = (
    os.environ.get("LOG_FILE")
    or globals().get("LOG_FILE")
    or "logs/post_analysis_summary.log"
)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Logging
logger = logging.getLogger("PostAnalysisLogger")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)

logger.info("=== Starting script to process & summarise analysis outputs ===")

# ------------------------------------------------------------------------
# 2. FILE PATHS
# ------------------------------------------------------------------------
TOKEN_LEVEL_ATTENTION_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, "token_level_attention.csv")
FRACTION_HARMFUL_CSV       = os.path.join(ANALYSIS_OUTPUT_DIR, "fraction_harmful_attention.csv")
HEAD_SPECIFIC_CSV          = os.path.join(ANALYSIS_OUTPUT_DIR, "head_specific_attention.csv")
TRAJECTORIES_CSV           = os.path.join(ANALYSIS_OUTPUT_DIR, "hidden_state_trajectories.csv")
REPEATED_SIM_CSV           = os.path.join(ANALYSIS_OUTPUT_DIR, "repeated_token_similarity.csv")
PCA_CSV_PATTERN            = os.path.join(ANALYSIS_OUTPUT_DIR, "pca_*.csv")  # e.g. pca_hack.csv
LOGIT_LENS_CSV             = os.path.join(ANALYSIS_OUTPUT_DIR, "logit_lens.csv")
MULTI_FILE_AGGREGATES_CSV  = os.path.join(ANALYSIS_OUTPUT_DIR, "multi_file_aggregates.csv")

# ------------------------------------------------------------------------
#                        1) TOKEN-LEVEL ATTENTION
# ------------------------------------------------------------------------
def summarize_token_level_attention():
    if not os.path.isfile(TOKEN_LEVEL_ATTENTION_CSV):
        logger.warning("No token_level_attention.csv found. Skipping.")
        return

    logger.info("Summarizing token_level_attention.csv ...")
    df = pd.read_csv(TOKEN_LEVEL_ATTENTION_CSV)

    min_val = df["avg_attention"].min()
    max_val = df["avg_attention"].max()
    mean_val = df["avg_attention"].mean()
    std_val = df["avg_attention"].std()

    logger.info(f"[Token-Level] Attention: min={min_val}, max={max_val}, mean={mean_val}, std={std_val}")

    # Plot histogram
    fig = plt.figure()
    plt.hist(df["avg_attention"], bins=50)
    plt.title("Distribution of avg_attention (Token-Level)")
    plt.xlabel("avg_attention")
    plt.ylabel("count")
    out_png = os.path.join(RESULTS_SUMMARY_DIR, "token_level_attention_hist.png")
    fig.savefig(out_png)
    plt.close(fig)
    logger.info(f"[SAVED] Token-level attention histogram -> {out_png}")

# ------------------------------------------------------------------------
#                  2) FRACTION OF ATTENTION ON HARMFUL WORDS
# ------------------------------------------------------------------------
def summarize_fraction_harmful():
    if not os.path.isfile(FRACTION_HARMFUL_CSV):
        logger.warning("No fraction_harmful_attention.csv found. Skipping.")
        return

    logger.info("Summarizing fraction_harmful_attention.csv ...")
    df = pd.read_csv(FRACTION_HARMFUL_CSV)

    # unify column name
    if "fraction_of_attention_on_harmful" in df.columns:
        df.rename(columns={"fraction_of_attention_on_harmful": "fraction_harmful"}, inplace=True)

    df = df.dropna(subset=["fraction_harmful"])

    min_val = df["fraction_harmful"].min()
    max_val = df["fraction_harmful"].max()
    mean_val = df["fraction_harmful"].mean()
    std_val = df["fraction_harmful"].std()

    logger.info(f"[Harmful Fraction] min={min_val}, max={max_val}, mean={mean_val}, std={std_val}")

    # Plot histogram
    fig = plt.figure()
    plt.hist(df["fraction_harmful"], bins=50)
    plt.title("Distribution of fraction of attention on harmful tokens")
    plt.xlabel("fraction_harmful")
    plt.ylabel("count")
    out_png = os.path.join(RESULTS_SUMMARY_DIR, "fraction_harmful_hist.png")
    fig.savefig(out_png)
    plt.close(fig)
    logger.info(f"[SAVED] Fraction-of-harmful histogram -> {out_png}")

# ------------------------------------------------------------------------
#            3) HEAD-SPECIFIC ATTENTION DISTRIBUTION
# ------------------------------------------------------------------------
def summarize_head_specific():
    if not os.path.isfile(HEAD_SPECIFIC_CSV):
        logger.warning("No head_specific_attention.csv found. Skipping.")
        return

    logger.info("Summarizing head_specific_attention.csv ...")
    df = pd.read_csv(HEAD_SPECIFIC_CSV)
    df = df.dropna(subset=["fraction_harmful"])

    # group by (layer, head)
    group = df.groupby(["layer", "head"])["fraction_harmful"].mean().reset_index()
    group_sorted = group.sort_values("fraction_harmful", ascending=False)

    top_20 = group_sorted.head(20)
    logger.info("Top 20 heads by average fraction_harmful:\n" + str(top_20))

    fig = plt.figure()
    x_positions = np.arange(len(top_20))
    plt.bar(x_positions, top_20["fraction_harmful"])
    plt.xticks(x_positions, [f"{r.layer}_head{r.head}" for r in top_20.itertuples()], rotation=90)
    plt.title("Top 20 (layer,head) by fraction_harmful")
    plt.xlabel("(layer, head)")
    plt.ylabel("mean fraction_harmful")
    plt.tight_layout()
    out_png = os.path.join(RESULTS_SUMMARY_DIR, "head_specific_top20.png")
    fig.savefig(out_png)
    plt.close(fig)
    logger.info(f"[SAVED] Head-specific distribution -> {out_png}")

# ------------------------------------------------------------------------
#         4) HIDDEN-STATE TRAJECTORIES + .npy FILES
# ------------------------------------------------------------------------
def summarize_hidden_trajectories():
    if not os.path.isfile(TRAJECTORIES_CSV):
        logger.warning("No hidden_state_trajectories.csv found. Skipping.")
        return

    logger.info("Summarising hidden_state_trajectories.csv + .npy files ...")
    rows = []
    with open(TRAJECTORIES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        logger.warning("No rows in hidden_state_trajectories.csv. Skipping.")
        return

    sample_size = 1000
    if len(rows) > sample_size:
        logger.info(f"Large trajectory dataset found ({len(rows)} rows). Sampling {sample_size}.")
        np.random.seed(42)
        rows = list(np.random.choice(rows, size=sample_size, replace=False))

    layer_norms_by_layer = {}
    for r in rows:
        npy_path = r["npy_path"]
        if not os.path.isfile(npy_path):
            continue
        traj = np.load(npy_path, allow_pickle=True).item()
        for layer_key, vec in traj.items():
            norm = float(np.linalg.norm(vec))
            layer_norms_by_layer.setdefault(layer_key, []).append(norm)

    layer_summary = []
    for layer_key, arr in layer_norms_by_layer.items():
        arr_np = np.array(arr)
        mean_ = arr_np.mean()
        std_ = arr_np.std()
        layer_summary.append((layer_key, mean_, std_, len(arr)))
    layer_summary.sort(key=lambda x: x[0])

    logger.info("** Hidden-State Norm Summary (sampled) **")
    for ls in layer_summary:
        logger.info(f"Layer={ls[0]}: mean_norm={ls[1]:.3f}, std={ls[2]:.3f}, count={ls[3]}")

    # line chart
    layer_indices = []
    means = []
    for lkey, mean_, std_, count_ in layer_summary:
        idx_str = lkey.replace("layer_", "")
        try:
            idx_val = int(idx_str)
        except:
            continue
        layer_indices.append(idx_val)
        means.append(mean_)

    fig = plt.figure()
    plt.plot(layer_indices, means, marker="o")
    plt.title("Avg Hidden-State Norm by Layer (sampled)")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Norm")
    out_png = os.path.join(RESULTS_SUMMARY_DIR, "hidden_state_norm_by_layer.png")
    fig.savefig(out_png)
    plt.close(fig)
    logger.info(f"[SAVED] Hidden-state norm by layer -> {out_png}")

# ------------------------------------------------------------------------
#          5) REPEATED TOKEN SIMILARITY
# ------------------------------------------------------------------------
def summarize_repeated_token_similarity():
    if not os.path.isfile(REPEATED_SIM_CSV):
        logger.warning("No repeated_token_similarity.csv found. Skipping.")
        return

    logger.info("Summarizing repeated_token_similarity.csv ...")
    df = pd.read_csv(REPEATED_SIM_CSV)
    df["layer_index"] = df["layer_key"].apply(lambda x: int(x.replace("layer_", "")) if "layer_" in x else -1)

    df_filtered = df[df["count_occurrences"] >= 2].copy()
    df_filtered.sort_values("avg_pairwise_cos_sim", ascending=False, inplace=True)
    top10 = df_filtered.head(10)
    logger.info("Top 10 repeated tokens by pairwise similarity (across all layers):\n" + str(top10))

    fig = plt.figure()
    plt.scatter(df_filtered["layer_index"], df_filtered["avg_pairwise_cos_sim"])
    plt.title("Layer vs. Average Pairwise Cosine Similarity (repeated tokens)")
    plt.xlabel("Layer Index")
    plt.ylabel("Avg Cosine Similarity")
    out_png = os.path.join(RESULTS_SUMMARY_DIR, "repeated_token_layer_vs_sim.png")
    fig.savefig(out_png)
    plt.close(fig)
    logger.info(f"[SAVED] Repeated-token similarity scatter -> {out_png}")

# ------------------------------------------------------------------------
#          6) PCA_{token}.csv
# ------------------------------------------------------------------------
def summarize_pca_files():
    pca_files = glob.glob(PCA_CSV_PATTERN)
    if not pca_files:
        logger.warning("No PCA CSV files found (pca_*.csv). Skipping.")
        return

    for pf in pca_files:
        token_name = os.path.basename(pf).replace("pca_", "").replace(".csv", "")
        logger.info(f"Summarizing PCA results for token '{token_name}' from file: {pf}")
        df = pd.read_csv(pf)

        fig = plt.figure()
        plt.scatter(df["pc1"], df["pc2"], s=10)
        plt.title(f"PCA for token '{token_name}' across layers/samples")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        out_png = os.path.join(RESULTS_SUMMARY_DIR, f"pca_{token_name}_scatter.png")
        fig.savefig(out_png)
        plt.close(fig)
        logger.info(f"[SAVED] PCA scatter for token='{token_name}' -> {out_png}")

# ------------------------------------------------------------------------
#          7) LOGIT LENS
# ------------------------------------------------------------------------
def summarize_logit_lens():
    if not os.path.isfile(LOGIT_LENS_CSV):
        logger.warning("No logit_lens.csv found. Skipping.")
        return

    logger.info("Summarizing logit_lens.csv ...")
    df = pd.read_csv(LOGIT_LENS_CSV)
    # columns: file_index, sample_index, layer, token_index, token_text, top_tokens

    # picking 5 random (file_index, sample_index, token_index) groups
    group_cols = ["file_index", "sample_index", "token_index"]
    grouped = df.groupby(group_cols)
    all_groups = list(grouped.groups.keys())
    sample_size = min(5, len(all_groups))

    # Use random.sample to avoid numpy's 1D array restriction:
    chosen_groups = random.sample(all_groups, sample_size)

    logger.info("** Example Logit Lens Evolution (up to 5 random tokens) **")
    for g in chosen_groups:
        subset = grouped.get_group(g).copy()

        def parse_layer(s):
            return int(s.replace("layer_", "")) if "layer_" in s else -1

        subset["layer_index"] = subset["layer"].apply(parse_layer)
        subset.sort_values("layer_index", inplace=True)
        token_txt = subset["token_text"].iloc[0]
        logger.info(f"For file={g[0]}, sample={g[1]}, token_index={g[2]}, token_text='{token_txt}'")
        for row in subset.itertuples():
            logger.info(f"  layer={row.layer} top_tokens={row.top_tokens}")

# ------------------------------------------------------------------------
#          8) MULTI-FILE AGGREGATES
# ------------------------------------------------------------------------
def summarize_multi_file_aggregates():
    if not os.path.isfile(MULTI_FILE_AGGREGATES_CSV):
        logger.warning("No multi_file_aggregates.csv found. Skipping.")
        return

    logger.info("Summarizing multi_file_aggregates.csv ...")
    df = pd.read_csv(MULTI_FILE_AGGREGATES_CSV)
    # columns: layer, mean_fraction_harmful, std_fraction_harmful, count

    def parse_layer_key(s):
        if isinstance(s, str) and "layer_" in s:
            return int(s.replace("layer_", ""))
        try:
            return int(s)
        except:
            return -1

    df["layer_index"] = df["layer"].apply(parse_layer_key)
    df.sort_values("layer_index", inplace=True)

    fig = plt.figure()
    plt.plot(df["layer_index"], df["mean_fraction_harmful"], marker="o")
    plt.title("Mean Fraction Harmful Attention by Layer (Aggregates)")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Fraction Harmful")
    out_png = os.path.join(RESULTS_SUMMARY_DIR, "multi_file_aggregates_fraction_harmful.png")
    fig.savefig(out_png)
    plt.close(fig)
    logger.info(f"[SAVED] Multi-file aggregates fraction harmful -> {out_png}")

    logger.info("Multi-file aggregate data:\n" + str(df))

# ------------------------------------------------------------------------
#                             MAIN PIPELINE
# ------------------------------------------------------------------------
def main():
    logger.info(f"Reading analysis outputs from: {ANALYSIS_OUTPUT_DIR}")
    logger.info(f"Will store summary results in: {RESULTS_SUMMARY_DIR}")

    summarize_token_level_attention()
    summarize_fraction_harmful()
    summarize_head_specific()
    summarize_hidden_trajectories()
    summarize_repeated_token_similarity()
    summarize_pca_files()
    summarize_logit_lens()
    summarize_multi_file_aggregates()

    logger.info("=== All post-analysis summaries complete! ===")

if __name__ == "__main__":
    main()
