import os
import json
import glob
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

##############################################################################
# 1) Logging Setup
##############################################################################
def init_logger(
    log_file="analyses/attention/attention_analysis.log",
    file_level=logging.DEBUG
):
    """
    Creates a logger that logs everything (DEBUG and up) to a file,
    but ONLY CRITICAL to the console.
    """
    # 1) Remove any existing handlers on the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Optionally set root logger to CRITICAL so it doesn't forward logs
    root_logger.setLevel(logging.CRITICAL)

    # 2) Create our named logger
    logger = logging.getLogger("ReNeLLMLogger")
    logger.setLevel(logging.DEBUG)  # internally handle all messages
    logger.propagate = False        # don't pass logs to root logger

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 3) File handler (captures everything)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(file_level)
    file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    # 4) Console handler (only CRITICAL)
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL)
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    logger.debug("Logger initialized. Console=CRITICAL, File=DEBUG.")
    return logger


##############################################################################
# 2) Helper Functions for Sublist/Substring Matching (includes fix for no leading space)
##############################################################################
def try_sublist_match(tokenizer, input_ids, phrase):
    """
    Sublist match of token IDs for `phrase` in `input_ids`.
    Returns a list of matched token positions if found.
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
        if (ch_s < end_char) and (ch_e > start_char):
            positions.append(i)
    return positions


def find_token_positions_flexible(tokenizer, input_ids, phrase, logger=None, log_prefix=""):
    """
    Attempt matches:
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


##############################################################################
# 3) Main Attention-Analysis Function
##############################################################################
def run_attention_analysis(
    prompt_key,
    base_dir="output/gemma-2-9b-it",
    main_prompts_file="data/renellm/full_levels.json",
    harmful_file="data/renellm/full_extracted_harmful.json",
    actionable_file="data/renellm/full_extracted_actionable.json",
    output_dir="analyses/attention",
    model_name="google/gemma-2-9b-it",
    logger=None
):
    """
    Analyzes how much attention is placed on the "harmful" substring
    and the "actionable" substring for a given `prompt_key`.

    1. Loads the main prompts (full_levels.json).
    2. Builds a mapping from element_id -> harmful and element_id -> actionable
       using the respective JSONs (full_extracted_harmful, full_extracted_actionable).
    3. Loads the .pt activation files from base_dir/<prompt_key>/activations_*.pt
       that were produced when running inference with the selected prompt_key.
    4. For each sample in the batch, uses `global_idx` to look up the correct row
       in the main prompts, retrieves that row's `element_id`,
       then obtains the harmful & actionable substrings from the respective
       extraction JSONs. Finally, measures the fraction of attention
       allocated to the matched tokens in each head of each layer.

    Outputs a CSV with columns:
        [folder, global_idx, element_id, layer, head, phrase_type, fraction_attention]

    where `phrase_type` is either "harmful" or "actionable".
    """
    # ------------------------------------------------------------------------------
    # 1) Setup logger if not provided
    # ------------------------------------------------------------------------------
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)

    logger.info(f"=== Starting attention analysis for prompt_key={prompt_key} ===")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------------------
    # 2) Load main prompts
    # ------------------------------------------------------------------------------
    if not os.path.isfile(main_prompts_file):
        raise FileNotFoundError(f"Could not find {main_prompts_file}")
    with open(main_prompts_file, "r", encoding="utf-8") as f:
        main_data = json.load(f)

    # ------------------------------------------------------------------------------
    # 3) Build dictionary for harmful_data: element_id -> { ... row data ... }
    # ------------------------------------------------------------------------------
    if not os.path.isfile(harmful_file):
        raise FileNotFoundError(f"Could not find {harmful_file}")
    with open(harmful_file, "r", encoding="utf-8") as f:
        harmful_data = json.load(f)

    harmful_map = {}
    for row in harmful_data:
        e_id = row["element_id"]
        harmful_map[e_id] = row

    # ------------------------------------------------------------------------------
    # 4) Build dictionary for actionable_data: element_id -> { ... row data ... }
    # ------------------------------------------------------------------------------
    if not os.path.isfile(actionable_file):
        raise FileNotFoundError(f"Could not find {actionable_file}")
    with open(actionable_file, "r", encoding="utf-8") as f:
        actionable_data = json.load(f)

    actionable_map = {}
    for row in actionable_data:
        e_id = row["element_id"]
        actionable_map[e_id] = row

    # ------------------------------------------------------------------------------
    # 5) Load the tokenizer
    # ------------------------------------------------------------------------------
    logger.info(f"Loading tokenizer for model={model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------------------
    # 6) Find the .pt files in base_dir/<prompt_key>/
    # ------------------------------------------------------------------------------
    folder_path = os.path.join(base_dir, prompt_key)
    pt_files = sorted(glob.glob(os.path.join(folder_path, "activations_*.pt")))
    logger.info(f"Found {len(pt_files)} .pt files in {folder_path}")
    if not pt_files:
        logger.warning("No .pt files found. Exiting.")
        return

    results_rows = []

    # ------------------------------------------------------------------------------
    # 7) Process each .pt file
    # ------------------------------------------------------------------------------
    for pt_file in tqdm(pt_files, desc=f"Folder={prompt_key}", ncols=80, disable=True):
        logger.debug(f"Loading .pt file: {pt_file}")
        data_dict = torch.load(pt_file, map_location="cpu")

        # 7a) Check we have attentions & input_ids
        all_attentions = data_dict.get("attentions", None)
        input_ids_tensor = data_dict.get("input_ids", None)
        if all_attentions is None or input_ids_tensor is None:
            logger.warning(f"File missing 'attentions' or 'input_ids': {pt_file}. Skipping.")
            continue

        # Convert each layer's attention to float32
        layer_names = sorted(all_attentions.keys(), key=lambda x: int(x.split("_")[-1]))
        attentions_list = []
        for lname in layer_names:
            attn_t = all_attentions[lname].to(torch.float32)
            attentions_list.append(attn_t)
        n_layers = len(attentions_list)

        # 7b) original_indices to map batch -> global_idx
        original_indices = data_dict.get("original_indices", None)
        if not original_indices:
            # fallback from filename if needed
            basename = os.path.basename(pt_file).replace(".pt", "")
            parts = basename.split("_")  # e.g. ["activations", "00000", "00004"]
            start_i = int(parts[1])
            logger.debug(f"No 'original_indices' found; fallback start_i={start_i}")

        batch_size, seq_len = input_ids_tensor.shape

        # 7c) For each sample in the batch
        for i_in_batch in range(batch_size):
            if original_indices:
                global_idx = original_indices[i_in_batch]
            else:
                global_idx = start_i + i_in_batch

            # Safety check
            if global_idx >= len(main_data):
                logger.debug(
                    f"global_idx={global_idx} >= main_data size={len(main_data)}. Skipping."
                )
                continue

            # Grab this row from the main prompt data
            prompt_row = main_data[global_idx]
            elem_id = prompt_row["element_id"]

            # If either map is missing the element_id, skip
            if elem_id not in harmful_map or elem_id not in actionable_map:
                logger.warning(f"Element {elem_id} not found in harmful/actionable maps. Skipping.")
                continue

            # 7d) Retrieve the harmful & actionable substrings for the chosen prompt_key
            harmful_str = harmful_map[elem_id].get(prompt_key, "")
            actionable_str = actionable_map[elem_id].get(prompt_key, "")

            # 7e) For each type ("harmful", "actionable"), find matched positions
            #     Then measure fraction of attention per layer/head
            for phrase_type, phrase_str in [("harmful", harmful_str),
                                            ("actionable", actionable_str)]:

                matched_positions = find_token_positions_flexible(
                    tokenizer,
                    input_ids_tensor[i_in_batch],
                    phrase_str,
                    logger=logger,
                    log_prefix=f"[glob_idx={global_idx}, {phrase_type}] "
                )

                # Now compute fraction of attention over matched tokens
                for layer_idx, attn_batch in enumerate(attentions_list):
                    attn_l = attn_batch[i_in_batch]  # shape [n_heads, seq_len, seq_len]
                    n_heads = attn_l.shape[0]
                    for h_idx in range(n_heads):
                        attn_head = attn_l[h_idx]

                        if not matched_positions:
                            # If no match, fraction=0
                            row = {
                                "folder": prompt_key,
                                "global_idx": global_idx,
                                "element_id": elem_id,
                                "layer": layer_idx,
                                "head": h_idx,
                                "phrase_type": phrase_type,
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
                                phrase_sum = attn_head[q, matched_positions].sum().item()
                                fraction_vals.append(phrase_sum / row_sum)
                        mean_fraction = float(np.mean(fraction_vals))

                        row = {
                            "folder": prompt_key,
                            "global_idx": global_idx,
                            "element_id": elem_id,
                            "layer": layer_idx,
                            "head": h_idx,
                            "phrase_type": phrase_type,
                            "fraction_attention": mean_fraction
                        }
                        results_rows.append(row)

    # ------------------------------------------------------------------------------
    # 8) Convert to DataFrame and save
    # ------------------------------------------------------------------------------
    df = pd.DataFrame(results_rows)
    out_csv = os.path.join(output_dir, f"attention_fractions_{prompt_key}.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(df)} rows to {out_csv}")
    logger.info("Done attention analysis.")
