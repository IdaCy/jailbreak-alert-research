import os
import glob
import warnings
import torch
import numpy as np
import time
import logging
from sklearn.decomposition import PCA

##############################################################################
# 1) OPTIONAL WARNINGS FILTERING
##############################################################################
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message="n_jobs value 1 overridden",
    category=UserWarning
)

##############################################################################
# 2) AGGREGATION LOGIC
##############################################################################
def aggregate_hidden_states(hs_tensor, method="last_token", last_few=5):
    """
    Aggregates hidden states into a single (batch_size, hidden_dim) vector.
    hs_tensor: shape (batch_size, seq_len, hidden_dim)
    method: "last_token", "last_few", or "mean_all"
    last_few: used if method="last_few"
    """
    if method == "last_token":
        return hs_tensor[:, -1, :]  # (batch_size, hidden_dim)
    elif method == "last_few":
        seq_len = hs_tensor.shape[1]
        if seq_len < last_few:
            return hs_tensor.mean(dim=1)
        else:
            return hs_tensor[:, -last_few:, :].mean(dim=1)
    elif method == "mean_all":
        return hs_tensor.mean(dim=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


##############################################################################
# 3) MAIN FUNCTION: Minimal analysis of .pt files
##############################################################################
def run_minimal_activations_process(
    prompt_type="attack",
    input_dir="output/extractions/gemma9bit/attack",
    output_dir="output/minimal_process_prompt_type/gemma9bit/attack",
    last_few_tokens=5,
    layers_of_interest=None,
    logger=None
):
    """
    Processes all .pt files for `prompt_type` in `input_dir`:
      1) Loads hidden_states from each file (the "hidden_states" dict).
      2) Aggregates them with 3 methods: "last_token", "last_few", "mean_all".
      3) Subsamples + runs PCA(2D) for quick dimension reduction.
      4) Saves final data in <output_dir>/analysis_results.pt

    :param prompt_type: String label, only for reference/logging.
    :param input_dir: Folder with .pt files like "activations_00000_00004.pt".
    :param output_dir: Where to store final "analysis_results.pt".
    :param last_few_tokens: If method="last_few", how many tokens to average.
    :param layers_of_interest: List of layer indices. Default=[0,5,10,15,20,25].
    :param logger: A logger object (optional). If None, minimal prints are used.
    """

    # 3a) If no logger is provided, create a minimal one that prints only warnings
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.WARNING)

    if layers_of_interest is None:
        layers_of_interest = [0, 5, 10, 15, 20, 25]

    # 3b) Basic logs about the run
    logger.info(f"=== Starting minimal activations analysis for '{prompt_type}' ===")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Last few tokens: {last_few_tokens}")
    logger.info(f"Layers of interest: {layers_of_interest}")

    os.makedirs(output_dir, exist_ok=True)

    # 3c) Gather .pt files
    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    logger.info(f"Found {len(pt_files)} .pt files in '{input_dir}'")
    if not pt_files:
        logger.warning("No .pt files found. Exiting.")
        return

    # aggregator_outputs[layer][method] = list of Tensors
    agg_methods = ["last_token", "last_few", "mean_all"]
    aggregator_outputs = {
        layer: {m: [] for m in agg_methods} for layer in layers_of_interest
    }

    # 3d) Load hidden states from each file
    for i, fpath in enumerate(pt_files, 1):
        logger.debug(f"Processing file #{i}: {os.path.basename(fpath)}")
        data = torch.load(fpath, map_location="cpu")
        hidden_states = data.get("hidden_states", None)
        if hidden_states is None:
            logger.warning(f"No 'hidden_states' in {fpath}, skipping.")
            continue

        # For each layer of interest, aggregate
        for layer in layers_of_interest:
            layer_key = f"layer_{layer}"
            if layer_key not in hidden_states:
                logger.debug(f"Missing {layer_key} in {fpath}, skipping.")
                continue

            hs_tensor = hidden_states[layer_key]  # shape (batch_size, seq_len, hidden_dim)

            for m in agg_methods:
                aggregated = aggregate_hidden_states(
                    hs_tensor, method=m, last_few=last_few_tokens
                )
                aggregator_outputs[layer][m].append(aggregated)

    # 3e) Concatenate all data for each layer/method, convert to float32 => numpy
    for layer in layers_of_interest:
        for m in agg_methods:
            stack_list = aggregator_outputs[layer][m]
            if len(stack_list) == 0:
                aggregator_outputs[layer][m] = np.zeros((0, 0), dtype=np.float32)
                continue
            cat_tensor = torch.cat(stack_list, dim=0)  # shape (N, hidden_dim)
            cat_tensor = cat_tensor.to(torch.float32)
            aggregator_outputs[layer][m] = cat_tensor.numpy()

    # 3f) Quick dimension reduction with PCA(2D). We'll do a small subsample (max 500)
    logger.info("Running PCA(2D) on each layer/method with <= 500 samples.")
    max_samples = 500
    pca_results = {}
    from sklearn.decomposition import PCA
    for layer in layers_of_interest:
        pca_results[layer] = {}
        for m in agg_methods:
            X = aggregator_outputs[layer][m]
            if X.shape[0] == 0:
                pca_results[layer][m] = np.zeros((0, 2), dtype=np.float32)
                continue

            # Subsample if needed
            N = X.shape[0]
            if N > max_samples:
                idx = np.random.choice(N, size=max_samples, replace=False)
                Xsub = X[idx, :]
            else:
                Xsub = X

            pca_2d = PCA(n_components=2, random_state=42).fit_transform(Xsub)
            pca_results[layer][m] = pca_2d

    # 3g) Save final results
    out_dict = {
        "aggregator_outputs": aggregator_outputs,  # aggregator arrays (N, hidden_dim)
        "pca_2d": pca_results,                     # subsampled PCA(2D) results
        "layers_of_interest": layers_of_interest,
        "agg_methods": agg_methods,
        "prompt_type": prompt_type,
        "last_few_tokens": last_few_tokens
    }
    out_path = os.path.join(output_dir, "analysis_results.pt")
    torch.save(out_dict, out_path)
    logger.info(f"Saved final results to {out_path}")
    logger.info("Minimal activations analysis complete.")
