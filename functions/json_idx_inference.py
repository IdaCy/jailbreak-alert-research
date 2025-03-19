import os
import torch
import logging
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def init_logger(
    log_file="logs/progress.log",
    error_log="logs/errors.log",
    console_level=logging.INFO,
    file_level=logging.DEBUG
):
    """
    Set up a logger with both console and file handlers.
    Returns the logger object.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(os.path.dirname(error_log), exist_ok=True)

    logger = logging.getLogger("ReNeLLMLogger")
    logger.setLevel(logging.DEBUG)  # master level

    # Remove any existing handlers (useful if re-initializing in a notebook)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Error file handler
    error_handler = logging.FileHandler(error_log, mode="a", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("Logger initialized.")
    return logger


def load_json_attacks(file_path, prompt_key="attack_jailbreak", max_samples=None, logger=None):
    """
    Reads a JSON file (which should be a list of objects).
    Returns a list of (orig_index, text) for each row that has a non-empty
    value for 'prompt_key'.
    """
    if logger:
        logger.debug(f"Loading JSON from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of objects in JSON, got type={type(data)}")

    filtered = []
    for i, item in enumerate(data):
        txt = item.get(prompt_key, "").strip()
        if txt:
            filtered.append((i, txt))  # store original index + text

    if max_samples is not None and max_samples < len(filtered):
        filtered = filtered[:max_samples]

    if logger:
        logger.info(f"Found {len(filtered)} lines in JSON with non-empty '{prompt_key}'.")
    return filtered


def load_model(
    MODEL_NAME="google/gemma-2-9b-it",
    USE_BFLOAT16=True,
    HF_TOKEN=None,
    logger=None
):
    """
    Loads a tokenizer and model from Hugging Face, using `device_map='auto'`.
    Returns (model, tokenizer).
    """
    if logger:
        logger.info(f"Loading tokenizer from {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_auth_token=HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if logger:
            logger.debug("No pad_token on tokenizer; using eos_token as pad.")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        max_memory = {0: f"{int(gpu_mem * 0.9)}GB"}
        if logger:
            logger.info(f"GPU is available. Setting max_memory={max_memory}")
    else:
        raise RuntimeError("No GPUs available! Ensure you are on a GPU runtime.")

    if logger:
        logger.info(f"Loading model {MODEL_NAME} with bfloat16={USE_BFLOAT16}, device_map='auto'")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
        use_auth_token=HF_TOKEN
    )
    model.eval()

    if logger:
        logger.info("Model loaded successfully.")
    return model, tokenizer


def run_inference_and_capture(
    model,
    tokenizer,
    text_batch,
    batch_index,
    MAX_SEQ_LENGTH=2048,
    EXTRACT_HIDDEN_LAYERS=[0, 5, 10, 15, 20, 25],
    EXTRACT_ATTENTION_LAYERS=[0, 5, 10, 15, 20, 25],
    TOP_K_LOGITS=10,
    logger=None,
    generation_kwargs=None
):
    """
    Runs a forward pass on the given `text_batch`, captures hidden states,
    attentions, and top-k logits. Also generates text with given generation
    parameters. Returns a dict of tensors and the final predictions.
    """
    if generation_kwargs is None:
        # Default generation parameters
        generation_kwargs = {
            "max_new_tokens": 100,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        }

    try:
        encodings = tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to("cuda")
        attention_mask = encodings["attention_mask"].to("cuda")

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )
            gen_out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )

        # Capture hidden states
        sel_hid = {}
        if outputs.hidden_states is not None:
            for l in EXTRACT_HIDDEN_LAYERS:
                if l < len(outputs.hidden_states):
                    sel_hid[f"layer_{l}"] = outputs.hidden_states[l].cpu().to(torch.bfloat16)

        # Capture attentions
        sel_attn = {}
        if outputs.attentions is not None:
            for l in EXTRACT_ATTENTION_LAYERS:
                if l < len(outputs.attentions):
                    sel_attn[f"layer_{l}"] = outputs.attentions[l].cpu().to(torch.bfloat16)

        # Capture top-k logits
        logits = outputs.logits
        topk_vals, topk_ids = torch.topk(logits, k=TOP_K_LOGITS, dim=-1)
        topk_vals = topk_vals.cpu().to(torch.bfloat16)
        topk_ids = topk_ids.cpu()

        # Decode generation
        final_preds = [
            tokenizer.decode(g, skip_special_tokens=True) for g in gen_out.cpu()
        ]

        return {
            "hidden_states": sel_hid,
            "attentions": sel_attn,
            "topk_logits": topk_vals,
            "topk_indices": topk_ids,
            "input_ids": input_ids.cpu(),
            "final_predictions": final_preds
        }
    except torch.cuda.OutOfMemoryError:
        if logger:
            logger.error(f"OOM at batch index {batch_index}, clearing cache.")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        if logger:
            logger.exception(f"Error at batch index {batch_index}: {e}")
        return None


def run_inference(
    model,
    tokenizer,
    data,
    BATCH_SIZE=4,
    OUTPUT_DIR="output/extractions",
    EXTRACT_HIDDEN_LAYERS=[0, 5, 10, 15, 20, 25],
    EXTRACT_ATTENTION_LAYERS=[0, 5, 10, 15, 20, 25],
    TOP_K_LOGITS=10,
    MAX_SEQ_LENGTH=2048,
    logger=None,
    generation_kwargs=None
):
    """
    Iterates over `data` (a list of (index, text)) in batches, runs inference
    and captures model activations and outputs. Saves .pt files to OUTPUT_DIR.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_indices = [r[0] for r in data]
    all_texts = [r[1] for r in data]
    total_samples = len(all_texts)

    if logger:
        logger.info(f"Clearing CUDA cache.")
    torch.cuda.empty_cache()

    if logger:
        logger.info("=== Starting main inference loop ===")

    for start_idx in range(0, total_samples, BATCH_SIZE):
        end_idx = start_idx + BATCH_SIZE
        batch_texts = all_texts[start_idx:end_idx]
        batch_indices = all_indices[start_idx:end_idx]

        if logger and (start_idx % 1000 == 0):
            logger.info(f"Processing batch {start_idx} / {total_samples}...")

        out_dict = run_inference_and_capture(
            model,
            tokenizer,
            batch_texts,
            batch_index=start_idx,
            MAX_SEQ_LENGTH=MAX_SEQ_LENGTH,
            EXTRACT_HIDDEN_LAYERS=EXTRACT_HIDDEN_LAYERS,
            EXTRACT_ATTENTION_LAYERS=EXTRACT_ATTENTION_LAYERS,
            TOP_K_LOGITS=TOP_K_LOGITS,
            logger=logger,
            generation_kwargs=generation_kwargs
        )

        if out_dict is not None:
            # We'll store the original indices in out_dict as well
            out_dict["original_indices"] = batch_indices
            # Save to disk
            filename = os.path.join(
                OUTPUT_DIR, f"activations_{start_idx:05d}_{end_idx:05d}.pt"
            )
            torch.save(out_dict, filename)

            if logger:
                logger.debug(f"Saved .pt to {filename}")

    if logger:
        logger.info("Inference complete.")
        logger.info(f"All .pt files are in {OUTPUT_DIR}.")
