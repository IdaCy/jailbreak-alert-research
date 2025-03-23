import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def init_eval_logger(log_file="analyses/evaluation/evaluation.log", level=logging.DEBUG):
    """
    Simple logger for the evaluation script. 
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("ReNeLLMEvalLogger")
    logger.setLevel(level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(level)
    file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    logger.debug("Evaluation logger initialized.")
    return logger


def load_all_attention_csvs(input_dir="analyses/attention", logger=None):
    """
    Loads all CSVs that match 'attention_fractions_*.csv' from `input_dir`.
    Merges them into a single dataframe with columns:
        ['folder', 'global_idx', 'element_id', 'layer', 'head', 'phrase_type', 'fraction_attention']
    Returns the merged dataframe.
    """
    if logger is None:
        logger = logging.getLogger("ReNeLLMEvalLogger")

    csv_files = sorted(glob.glob(os.path.join(input_dir, "attention_fractions_*.csv")))
    logger.info(f"Found {len(csv_files)} CSV files in {input_dir}")

    all_dfs = []
    for csv_path in csv_files:
        # Example file name: attention_fractions_<PROMPT_TYPE>.csv
        logger.debug(f"Loading {csv_path}")
        df_chunk = pd.read_csv(csv_path)
        all_dfs.append(df_chunk)

    if not all_dfs:
        logger.warning(f"No CSV data found in {input_dir}. Returning empty dataframe.")
        return pd.DataFrame()

    merged_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Merged {len(csv_files)} files -> total rows: {len(merged_df)}")
    return merged_df


def compute_summary_stats(df, logger=None):
    """
    Computes a variety of groupby-level summary stats on the combined DataFrame.
    Returns a dictionary of dataframes (or Series) with different breakdowns.
    """
    if logger is None:
        logger = logging.getLogger("ReNeLLMEvalLogger")

    summaries = {}

    # 1) Group by [folder, phrase_type, layer, head]
    #    We'll compute count, mean, min, max, std of fraction_attention
    grouping1 = ["folder", "phrase_type", "layer", "head"]
    stats1 = (df.groupby(grouping1)["fraction_attention"]
                .agg(["count", "mean", "min", "max", "std"])
                .reset_index())
    summaries["by_folder_phrase_layer_head"] = stats1

    # 2) Group by [folder, phrase_type, layer], average over heads
    grouping2 = ["folder", "phrase_type", "layer"]
    stats2 = (df.groupby(grouping2)["fraction_attention"]
                .agg(["count", "mean", "min", "max", "std"])
                .reset_index())
    summaries["by_folder_phrase_layer"] = stats2

    # 3) Group by [folder, phrase_type], average over layers & heads
    grouping3 = ["folder", "phrase_type"]
    stats3 = (df.groupby(grouping3)["fraction_attention"]
                .agg(["count", "mean", "min", "max", "std"])
                .reset_index())
    summaries["by_folder_phrase"] = stats3

    # 4) Overall average across everything, just to see a quick global statistic
    global_mean = df["fraction_attention"].mean()
    global_std = df["fraction_attention"].std()
    logger.info(f"Global average of fraction_attention = {global_mean:.4f} ± {global_std:.4f}")

    return summaries


def plot_attention_by_layer(df, output_dir="analyses/evaluation/plots", logger=None):
    """
    Creates and saves some example plots:
      - A lineplot of mean fraction_attention by layer, split by phrase_type and folder.
      - Possibly a heatmap of (layer x head) for each folder & phrase_type.
    """
    if logger is None:
        logger = logging.getLogger("ReNeLLMEvalLogger")

    os.makedirs(output_dir, exist_ok=True)

    # Example 1: lineplot of fraction_attention by layer, aggregated over heads
    # We'll group by [folder, phrase_type, layer] then plot
    grp = (df.groupby(["folder", "phrase_type", "layer"])["fraction_attention"]
             .mean()
             .reset_index())

    # We'll do a separate figure for each prompt folder
    folders = grp["folder"].unique()
    for folder in folders:
        subset = grp[grp["folder"] == folder]
        plt.figure(figsize=(7,5))
        sns.lineplot(
            data=subset,
            x="layer",
            y="fraction_attention",
            hue="phrase_type",
            marker="o"
        )
        plt.title(f"Mean Attention Fraction by Layer\n(folder={folder})")
        plt.ylim(0, None)
        plt.legend(title="Phrase Type")
        plt.tight_layout()
        outpath = os.path.join(output_dir, f"lineplot_by_layer_{folder}.png")
        plt.savefig(outpath)
        plt.close()
        logger.debug(f"Saved lineplot to {outpath}")

    # Example 2: Heatmap of mean fraction_attention over queries, by layer & head
    # We'll group by [folder, phrase_type, layer, head]
    grp2 = (df.groupby(["folder", "phrase_type", "layer", "head"])["fraction_attention"]
              .mean()
              .reset_index())

    # For each folder + phrase_type combination, produce a heatmap
    for folder in folders:
        f_subset = grp2[grp2["folder"] == folder]
        phrase_types = f_subset["phrase_type"].unique()

        for pt in phrase_types:
            sub = f_subset[f_subset["phrase_type"] == pt]
            # We pivot so that rows=layer, cols=head, values=mean fraction
            pivoted = sub.pivot(index="layer", columns="head", values="fraction_attention")

            plt.figure(figsize=(8, 6))
            sns.heatmap(pivoted, annot=False, cmap="Blues", cbar=True)
            plt.title(f"Heatmap: fraction_attention\nfolder={folder}, phrase_type={pt}")
            plt.ylabel("Layer")
            plt.xlabel("Head")
            plt.tight_layout()
            outpath = os.path.join(output_dir, f"heatmap_{folder}_{pt}.png")
            plt.savefig(outpath)
            plt.close()
            logger.debug(f"Saved heatmap to {outpath}")


def evaluate_all_results(
    input_dir="analyses/attention",
    output_dir="analyses/evaluation",
    save_summaries=True,
    logger=None
):
    """
    Main entry point to:
      1) load all attention_fractions_*.csv
      2) compute summary stats
      3) create example plots
      4) optionally save summary stats to CSV
    """
    if logger is None:
        logger = init_eval_logger(
            log_file=os.path.join(output_dir, "evaluation.log"),
            level=logging.DEBUG
        )
    else:
        os.makedirs(output_dir, exist_ok=True)

    # 1) Load all CSVs
    df = load_all_attention_csvs(input_dir=input_dir, logger=logger)
    if df.empty:
        logger.warning("No data found. Exiting evaluate_all_results.")
        return

    # 2) Compute summary stats
    summaries = compute_summary_stats(df, logger=logger)

    # 3) (Optional) Save summaries to CSV
    if save_summaries:
        for name, sdf in summaries.items():
            csv_name = f"{name}.csv"
            out_path = os.path.join(output_dir, csv_name)
            sdf.to_csv(out_path, index=False)
            logger.info(f"Saved summary {name} -> {out_path}")

    # 4) Plot & save
    plot_dir = os.path.join(output_dir, "plots")
    plot_attention_by_layer(df, output_dir=plot_dir, logger=logger)

    logger.info("All evaluations complete!")
