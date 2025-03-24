# file: functions/json_idx_inference.py

import os
import math
import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def init_logger(
    log_file="logs/inference.log",
    console_level=logging.INFO,
    file_level=logging.DEBUG
):
    """
    Creates a logger that writes detailed logs to a file
    and a more concise output to the console.
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.CRITICAL)  # no propagation

    logger = logging.getLogger("ReNeLLMLogger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # File handler (debug level)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(file_level)
    file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    logger.info("Logger initialized.")
    return logger


def load_model(
    model_name="google/gemma-2-9b-it",
    use_bfloat16=True,
    hf_token=None,
    logger=None
):
    """
    Loads the specified model and tokenizer from Hugging Face.
    """
    if logger is None:
        logger = logging.getLogger("ReNeLLMLogger")

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. Please enable a GPU in Colab.")

    logger.info(f"Loading tokenizer from '{model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("No pad_token found; using eos_token as pad_token.")

    gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    max_memory = {0: f"{int(gpu_mem*0.9)}GB"}
    logger.info(f"Loading model '{model_name}' (bfloat16={use_bfloat16}) with device_map=auto")

    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
        use_auth_token=hf_token
    )
    model.eval()
    logger.info("Model loaded successfully.")
    return model, tokenizer


def load_json_attacks(file_path, prompt_key="attack", max_samples=None, logger=None):
    """
    Load a JSON file (a list of objects). Return a list of (original_idx, text)
    for each row that has a non-empty value for 'prompt_key'.
    """
    if logger is None:
        logger = logging.getLogger("ReNeLLMLogger")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Could not find JSON file: {file_path}")

    logger.debug(f"Reading JSON from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of objects.")

    results = []
    for i, row in enumerate(data):
        text = row.get(prompt_key, "").strip()
        if text:
            results.append((i, text))

    if max_samples is not None and max_samples < len(results):
        results = results[:max_samples]

    logger.info(f"Loaded {len(results)} items from '{file_path}' with prompt_key='{prompt_key}'.")
    return results


def run_inf(
    model,
    tokenizer,
    data,
    output_dir="output/",
    batch_size=4,
    max_seq_length=2048,
    extract_hidden_layers=None,
    extract_attention_layers=None,
    top_k_logits=10,
    logger=None,
    generation_kwargs=None
):
    """
    data is a list of (orig_index, text). We'll run inference capturing:
       - hidden states (selected layers)
       - attentions (selected layers)
       - top-k logits
       - final predictions

    We'll store each batch's results in a .pt file with fields:
       - 'hidden_states', 'attentions', 'topk_vals', 'topk_indices',
         'input_ids', 'final_predictions', 'original_indices'

    This way the attention analysis can rely on 'original_indices' to align to the JSON.
    """
    if logger is None:
        logger = logging.getLogger("ReNeLLMLogger")

    if extract_hidden_layers is None:
        extract_hidden_layers = [0, 5, 10, 15]
    if extract_attention_layers is None:
        extract_attention_layers = [0, 5, 10, 15]

    os.makedirs(output_dir, exist_ok=True)

    # Clear GPU cache
    logger.info("Clearing CUDA cache before starting.")
    torch.cuda.empty_cache()

    if generation_kwargs is None:
        generation_kwargs = {
            "do_sample": True,
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        }

    total_samples = len(data)
    total_batches = math.ceil(total_samples / batch_size)
    logger.info(f"=== Starting inference. #samples={total_samples}, batch_size={batch_size} ===")

    for batch_idx in range(total_batches):
        start_i = batch_idx * batch_size
        end_i = min((batch_idx + 1) * batch_size, total_samples)
        batch_items = data[start_i:end_i]
        batch_indices = [x[0] for x in batch_items]
        batch_texts = [x[1] for x in batch_items]

        if batch_idx % 20 == 0:  # log progress every ~20 batches
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} (samples {start_i}-{end_i-1})")

        # Tokenize
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].cuda()
        attention_mask = encodings["attention_mask"].cuda()

        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True
                )
                # Generate text
                gen_out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generation_kwargs
                )

            # Collect hidden states
            hidden_map = {}
            for layer_idx in extract_hidden_layers:
                if layer_idx < len(outputs.hidden_states):
                    # Convert to smaller dtype if you like
                    hidden_map[f"layer_{layer_idx}"] = outputs.hidden_states[layer_idx].cpu()

            # Collect attentions
            attn_map = {}
            for layer_idx in extract_attention_layers:
                if layer_idx < len(outputs.attentions):
                    attn_map[f"layer_{layer_idx}"] = outputs.attentions[layer_idx].cpu()

            # Top-k
            logits = outputs.logits  # shape [batch, seq_len, vocab_size]
            topk_vals, topk_indices = torch.topk(logits, k=top_k_logits, dim=-1)
            topk_vals = topk_vals.cpu()
            topk_indices = topk_indices.cpu()

            # Final predictions
            decoded_preds = [
                tokenizer.decode(o, skip_special_tokens=True) for o in gen_out.cpu()
            ]

            out_dict = {
                "hidden_states": hidden_map,
                "attentions": attn_map,
                "topk_vals": topk_vals,
                "topk_indices": topk_indices,
                "input_ids": input_ids.cpu(),
                "final_predictions": decoded_preds,
                "original_indices": batch_indices  # CRUCIAL for alignment in analysis
            }

            save_name = f"activations_{start_i:05d}_{end_i:05d}.pt"
            save_path = os.path.join(output_dir, save_name)
            torch.save(out_dict, save_path)
            logger.debug(f"Saved batch => {save_path}")

        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM error on batch {batch_idx}. Clearing cache and continuing.")
            torch.cuda.empty_cache()
        except Exception as ex:
            logger.exception(f"Error on batch {batch_idx}: {ex}")

    logger.info("=== Inference Complete ===")
