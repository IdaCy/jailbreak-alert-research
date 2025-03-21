import os
import json
import glob
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

def init_logger(
    log_file="analyses/3e_results_attention/attention_analysis.log",
    file_level=logging.DEBUG
):
    """
    Creates a logger that logs everything (DEBUG and up) to a file,
    but ONLY CRITICAL to the console. This is as minimal as you can get
    without outright disabling logging.
    """
    # 1) Remove any existing handlers on the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.CRITICAL)

    # 2) Create our named logger
    logger = logging.getLogger("ReNeLLMLogger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(file_level)
    file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    # Console handler (critical-only)
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL)
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    logger.debug("Logger initialized. Console=CRITICAL, File=DEBUG.")
    return logger


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
        # Overlap check
        if ch_s is not None and ch_e is not None:
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

    # 3) substring offset approach
    sub_pos = substring_offset_mapping(tokenizer, input_ids, phrase)
    if logger:
        logger.debug(f"{log_prefix}Substring fallback -> positions={sub_pos}")
    return sub_pos


def compute_fraction_of_attention(attn_head, matched_positions):
    """
    For a single attention head (shape [seq_len, seq_len]) and
    a list of matched token positions, compute the average fraction
    of attention that each token in the sequence places on those matched positions.
    """
    if not matched_positions:
        return 0.0
    seq_len = attn_head.shape[0]
    fraction_vals = []
    for q in range(seq_len):
        row_sum = attn_head[q].sum().item()
        if row_sum <= 0:
            fraction_vals.append(0.0)
        else:
            phrase_sum = attn_head[q, matched_positions].sum().item()
            fraction_vals.append(phrase_sum / row_sum)
    return float(np.mean(fraction_vals))


def run_attention_analysis(
    prompt_type="attack",
    base_dir="output/gemma-2-9b-it",
    prompts_file="data/renellm/full_levels.json",
    harmful_file="data/renellm/full_extracted_harmful.json",
    actionable_file="data/renellm/full_extracted_actionable.json",
    output_dir="analyses/3e_results_attention",
    model_name="google/gemma-2-9b-it",
    logger=None
):
    """
    Analyzes how much attention the model pays to a *harmful substring*
    and an *actionable substring*, for a single prompt type, using
    previously saved .pt files from inference.

    Steps:
      1) Load the prompt data (prompts_file), which includes `element_id` and the text used for inference.
      2) Build dictionaries from harmful_file and actionable_file by `element_id`.
      3) For each .pt file in base_dir/prompt_type, find the 'attentions', 'input_ids', and 'original_indices'.
         - Identify the matching row in the main prompts data to retrieve `element_id`.
         - Retrieve the harmful/actionable substrings from their respective JSONs for that same prompt_type.
         - Find positions in the input_ids (via flexible sublist/substring matching).
         - Compute fraction of attention for each (harmful & actionable).
      4) Save a CSV with columns:
         [folder, global_idx, element_id, layer, head,
          fraction_harmful, fraction_actionable]
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.WARNING)

    logger.info(f"=== Starting attention analysis for PROMPT_TYPE={prompt_type} ===")
    os.makedirs(output_dir, exist_ok=True)

    # (A) Load main prompts data
    if not os.path.isfile(prompts_file):
        raise FileNotFoundError(f"Could not find prompts_file at {prompts_file}")
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
    logger.info(f"Loaded {len(prompts_data)} rows from prompts_file.")

    # We'll build a quick list -> by index we get the row
    # So 'element_id' = prompts_data[idx]["element_id"]
    # and the text for prompt_type is prompts_data[idx][prompt_type] (the original prompt)

    # (B) Load harmful data (build a dict from element_id -> row)
    if not os.path.isfile(harmful_file):
        raise FileNotFoundError(f"Could not find harmful_file at {harmful_file}")
    with open(harmful_file, "r", encoding="utf-8") as f:
        harmful_list = json.load(f)
    harmful_map = {}
    for row in harmful_list:
        eid = row["element_id"]
        harmful_map[eid] = row
    logger.info(f"Loaded {len(harmful_map)} rows from harmful_file.")

    # (C) Load actionable data (dict from element_id -> row)
    if not os.path.isfile(actionable_file):
        raise FileNotFoundError(f"Could not find actionable_file at {actionable_file}")
    with open(actionable_file, "r", encoding="utf-8") as f:
        actionable_list = json.load(f)
    actionable_map = {}
    for row in actionable_list:
        eid = row["element_id"]
        actionable_map[eid] = row
    logger.info(f"Loaded {len(actionable_map)} rows from actionable_file.")

    # (D) Gather the .pt files
    folder_path = os.path.join(base_dir, prompt_type)
    pt_files = sorted(glob.glob(os.path.join(folder_path, "activations_*.pt")))
    logger.info(f"Found {len(pt_files)} .pt files in '{folder_path}'.")
    if not pt_files:
        logger.warning("No .pt files found. Exiting.")
        return

    # (E) Process each .pt
    results_rows = []
    for pt_file in tqdm(pt_files, desc=f"Folder={prompt_type}", disable=True):
        logger.debug(f"Loading .pt file: {pt_file}")
        data_dict = torch.load(pt_file, map_location="cpu")
        all_attentions = data_dict.get("attentions", None)
        input_ids_tensor = data_dict.get("input_ids", None)
        original_indices = data_dict.get("original_indices", None)

        if all_attentions is None or input_ids_tensor is None or original_indices is None:
            logger.warning(f"Missing some keys in {pt_file} => skipping.")
            continue

        # Convert each layer to float32
        layer_names = sorted(all_attentions.keys(), key=lambda x: int(x.split("_")[-1]))
        attentions_list = [all_attentions[l].to(torch.float32) for l in layer_names]
        n_layers = len(attentions_list)

        batch_size, seq_len = input_ids_tensor.shape

        # For each sample in the batch
        for i_in_batch in range(batch_size):
            global_idx = original_indices[i_in_batch]
            # In the "prompts_file", the row is prompts_data[global_idx]
            # So we read element_id from that
            if global_idx >= len(prompts_data):
                logger.debug(f"global_idx={global_idx} >= {len(prompts_data)} => skipping.")
                continue

            prompt_row = prompts_data[global_idx]
            element_id = prompt_row.get("element_id", None)
            if element_id is None:
                logger.debug(f"No element_id in row {global_idx}? skipping.")
                continue

            # Grab the harmful substring
            # e.g. harmful_map[element_id][prompt_type]
            harmful_str = ""
            actionable_str = ""
            if element_id in harmful_map:
                harmful_str = harmful_map[element_id].get(prompt_type, "").strip()
            else:
                logger.debug(f"element_id={element_id} not in harmful_map? skipping harmful part.")
            if element_id in actionable_map:
                actionable_str = actionable_map[element_id].get(prompt_type, "").strip()
            else:
                logger.debug(f"element_id={element_id} not in actionable_map? skipping actionable part.")

            # Token positions
            input_ids = input_ids_tensor[i_in_batch]

            # Find positions for harmful_str
            harmful_positions = find_token_positions_flexible(
                tokenizer, input_ids, harmful_str, logger=logger,
                log_prefix=f"[global_idx={global_idx}, harmful]"
            )
            # Find positions for actionable_str
            actionable_positions = find_token_positions_flexible(
                tokenizer, input_ids, actionable_str, logger=logger,
                log_prefix=f"[global_idx={global_idx}, actionable]"
            )

            # For each layer & head, compute fraction
            for layer_idx, attn_batch in enumerate(attentions_list):
                attn_l = attn_batch[i_in_batch]  # shape [n_heads, seq_len, seq_len]
                n_heads = attn_l.shape[0]
                for h_idx in range(n_heads):
                    attn_head = attn_l[h_idx]
                    fraction_harmful = compute_fraction_of_attention(attn_head, harmful_positions)
                    fraction_actionable = compute_fraction_of_attention(attn_head, actionable_positions)

                    row = {
                        "folder": prompt_type,
                        "global_idx": global_idx,
                        "element_id": element_id,
                        "layer": layer_idx,
                        "head": h_idx,
                        "fraction_harmful": fraction_harmful,
                        "fraction_actionable": fraction_actionable
                    }
                    results_rows.append(row)

    # (F) Save final CSV
    df = pd.DataFrame(results_rows)
    out_csv = os.path.join(output_dir, f"attention_fractions_{prompt_type}.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(df)} rows to '{out_csv}'.")
    logger.info("Done attention analysis.")
