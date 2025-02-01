import os
import json
import glob
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# ----------------------------------------------------------------------
# 1. Optional: A dictionary to define which keys to look for, per prompt_type
# ----------------------------------------------------------------------
FOLDER_TO_KEYS = {
    "attack": [
        "harm_attack",
        "action_attack",
    ],
    "attack_jailbreak": [
        "harm_jailbreak",
        "action_jailbreak",
        "context_attack_jailbreak",
    ],
    "jailbreak_stronger": [
        "harm_jailbreak",
        "action_jailbreak",
        "context_jailbreak_stronger"
    ],
    "jailbreak_weaker": [
        "harm_jailbreak",
        "action_jailbreak",
        "context_jailbreak_weaker"
    ],
    "jailbreak_shorter": [
        "harm_jailbreak",
        "action_jailbreak",
        "context_jailbreak_shorter"
    ],
}


# ----------------------------------------------------------------------
# 2. Logging Setup
# ----------------------------------------------------------------------
def init_logger(
    log_file="analyses/3e_results_attention/attention_analysis.log",
    console_level=logging.WARNING,  # only warnings/errors to console
    file_level=logging.DEBUG
):
    """
    Set up a file+console logger. Returns the logger object.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Master level (allows all messages up to DEBUG internally)

    # Remove any existing handlers (helpful if re-initializing in a notebook)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler (full debug logs to file)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(file_level)
    fh_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler (only warnings/errors to console)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # Minimal console message
    logger.info("Logger initialized for attention analysis (console=WARN, file=DEBUG).")

    return logger


# ----------------------------------------------------------------------
# 3. Helper Functions for Sublist/Substring Matching
# ----------------------------------------------------------------------
def try_sublist_match(tokenizer, input_ids, phrase):
    """
    Sublist match of token IDs for `phrase` in `input_ids`.
    Returns a list of matched token positions.
    """
    phrase_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    example_id_list = input_ids.tolist()
    matches = []
    for start_i in range(len(example_id_list) - len(phrase_ids) + 1):
        window = example_id_list[start_i : start_i + len(phrase_ids)]
        if window == phrase_ids:
            matches.extend(range(start_i, start_i + len(phrase_ids)))
    return sorted(set(matches))


def substring_offset_mapping(tokenizer, input_ids, phrase):
    """
    Decodes input_ids into text, finds substring positions of `phrase`,
    then uses offset mappings to translate back into token positions.
    Returns a list of matched token indices.
    """
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    idx = decoded_text.find(phrase)
    if idx == -1:
        return []
    encoded = tokenizer(decoded_text, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    start_char = idx
    end_char = idx + len(phrase)

    positions = []
    for i, (ch_s, ch_e) in enumerate(offsets):
        if ch_s is None or ch_e is None:
            continue
        # Overlap check
        if ch_s < end_char and ch_e > start_char:
            positions.append(i)
    return positions


def find_token_positions_flexible(tokenizer, input_ids, phrase, logger=None, log_prefix=""):
    """
    Attempt 3 matches:
      1) direct sublist
      2) leading-space sublist
      3) substring offset approach
    """
    phrase = phrase.strip()
    if not phrase:
        if logger:
            logger.debug(f"{log_prefix}Empty phrase => no match.")
        return []

    # 1) direct sublist
    direct = try_sublist_match(tokenizer, input_ids, phrase)
    if direct:
        if logger:
            logger.debug(f"{log_prefix}Direct sublist -> positions={direct}")
        return direct

    # 2) leading space sublist
    lead_phrase = " " + phrase
    lead = try_sublist_match(tokenizer, input_ids, lead_phrase)
    if lead:
        if logger:
            logger.debug(f"{log_prefix}Leading-space sublist -> positions={lead}")
        return lead

    # 3) substring approach
    sub_pos = substring_offset_mapping(tokenizer, input_ids, phrase)
    if logger:
        logger.debug(f"{log_prefix}Substring fallback -> positions={sub_pos}")
    return sub_pos


# ----------------------------------------------------------------------
# 4. Main Function to Run Attention Analysis
# ----------------------------------------------------------------------
def run_attention_analysis(
    prompt_type="attack",
    base_dir="output/extractions/gemma9bit",
    extracted_strings_file="data/renellm/extracted_strings.json",
    output_dir="analyses/3e_results_attention",
    model_name="google/gemma-2-9b-it",
    logger=None
):
    """
    Processes all .pt files in `base_dir/prompt_type` looking for
    'attentions' and 'input_ids'. Then:
      1. Loads phrases from `extracted_strings_file`
      2. Matches relevant tokens based on `prompt_type`
      3. Computes fraction of attention spent on the matched tokens
      4. Saves CSV results to `output_dir`
    """

    # If no logger given, build a minimal one
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.WARNING)

    logger.info(f"=== Starting attention analysis for PROMPT_TYPE={prompt_type} ===")

    os.makedirs(output_dir, exist_ok=True)

    # 4a) Load your extracted strings
    if not os.path.isfile(extracted_strings_file):
        raise FileNotFoundError(f"Could not find extracted_strings.json at {extracted_strings_file}")

    with open(extracted_strings_file, "r", encoding="utf-8") as f:
        extracted_data = json.load(f)

    # 4b) Load tokenizer
    logger.info(f"Loading tokenizer for model={model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4c) Gather .pt files for this folder
    folder_path = os.path.join(base_dir, prompt_type)
    pt_files = sorted(glob.glob(os.path.join(folder_path, "activations_*.pt")))
    logger.info(f"Found {len(pt_files)} .pt files in {folder_path}")
    if not pt_files:
        logger.warning("No .pt files found. Exiting.")
        return

    # 4d) Figure out which keys matter for this prompt_type
    relevant_keys = FOLDER_TO_KEYS.get(prompt_type, [])
    logger.info(f"Relevant phrase keys for '{prompt_type}': {relevant_keys}")

    results_rows = []

    # 4e) Process each .pt file
    # Disable tqdm output
    for pt_file in tqdm(pt_files, desc=f"Folder={prompt_type}", ncols=80, disable=True):
        logger.debug(f"Loading .pt file: {pt_file}")
        data_dict = torch.load(pt_file, map_location="cpu")

        # i) Check we have attentions and input_ids
        all_attentions = data_dict.get("attentions", None)
        input_ids_tensor = data_dict.get("input_ids", None)
        if all_attentions is None or input_ids_tensor is None:
            logger.warning(f"File missing 'attentions' or 'input_ids': {pt_file}. Skipping.")
            continue

        # ii) Convert each layer's attention to float32
        layer_names = sorted(all_attentions.keys(), key=lambda x: int(x.split("_")[-1]))
        attentions_list = []
        for lname in layer_names:
            attn_t = all_attentions[lname].to(torch.float32)
            attentions_list.append(attn_t)
        n_layers = len(attentions_list)
        batch_size, seq_len = input_ids_tensor.shape

        # iii) Determine global indices from .pt
        original_indices = data_dict.get("original_indices", None)
        if not original_indices:
            # fallback from filename
            basename = os.path.basename(pt_file).replace(".pt", "")
            parts = basename.split("_")  # e.g. ["activations","00000","00004"]
            start_i = int(parts[1])
            logger.debug(f"Fallback indexing: using filename-based start_i={start_i}")

        # iv) For each sample in the batch
        for i_in_batch in range(batch_size):
            if original_indices:
                global_idx = original_indices[i_in_batch]
            else:
                global_idx = start_i + i_in_batch

            if global_idx >= len(extracted_data):
                logger.debug(f"Global idx={global_idx} > extracted_data size={len(extracted_data)}. Skipping.")
                continue

            # We'll only match the keys relevant to this folder
            phrase_map = extracted_data[global_idx]

            # v) For each relevant key, find matched positions
            phrase_positions_map = {}
            for pk in relevant_keys:
                phrase_str = phrase_map.get(pk, "")
                log_prefix = f"[glob_idx={global_idx}, key={pk}] "
                matched_positions = find_token_positions_flexible(
                    tokenizer,
                    input_ids_tensor[i_in_batch],
                    phrase_str,
                    logger=logger,
                    log_prefix=log_prefix
                )
                phrase_positions_map[pk] = matched_positions

            # vi) Compute fraction of attention
            # shape of each attn_l[i_in_batch] is [n_heads, seq_len, seq_len]
            for layer_idx, attn_batch in enumerate(attentions_list):
                attn_l = attn_batch[i_in_batch]  # shape [n_heads, seq_len, seq_len]
                n_heads = attn_l.shape[0]
                for h_idx in range(n_heads):
                    attn_head = attn_l[h_idx]
                    for pk in relevant_keys:
                        matched_pos = phrase_positions_map[pk]
                        if not matched_pos:
                            # no tokens matched => fraction=0
                            row = {
                                "folder": prompt_type,
                                "global_idx": global_idx,
                                "layer": layer_idx,
                                "head": h_idx,
                                "phrase_key": pk,
                                "fraction_attention": 0.0
                            }
                            results_rows.append(row)
                            continue

                        fraction_vals = []
                        for q in range(seq_len):
                            row_sum = attn_head[q].sum().item()
                            if row_sum <= 0:
                                fraction_vals.append(0.0)
                            else:
                                phrase_sum = attn_head[q, matched_pos].sum().item()
                                fraction_vals.append(phrase_sum / row_sum)
                        mean_fraction = float(np.mean(fraction_vals))

                        row = {
                            "folder": prompt_type,
                            "global_idx": global_idx,
                            "layer": layer_idx,
                            "head": h_idx,
                            "phrase_key": pk,
                            "fraction_attention": mean_fraction
                        }
                        results_rows.append(row)

    # 4f) Convert to DataFrame and save
    df = pd.DataFrame(results_rows)
    out_csv = os.path.join(output_dir, f"attention_fractions_{prompt_type}.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(df)} rows to {out_csv}")
    logger.info("Done attention analysis.")
