# my_inference.py

import os
import torch
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(
    model_name="google/gemma-2-2b",
    use_bfloat16=True,
    hf_token=None,
    max_seq_length=2048,
    log_file="logs/model_load.log",
):
    """
    Loads the model & tokenizer exactly once. 
    Returns (model, tokenizer).
    """
    # Setup logging, or rely on your notebook’s logging
    logger = logging.getLogger("MyInferenceLoader")
    logger.setLevel(logging.INFO)

    # Possibly set up file handlers if you like:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info("=== load_model_and_tokenizer called ===")
    logger.info(f"model_name={model_name}, use_bfloat16={use_bfloat16}")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model...")
    torch.cuda.empty_cache()
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available.")
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        low_cpu_mem_usage=True,
        device_map=device_map,
        use_auth_token=hf_token
    )
    model.eval()

    logger.info("Model + tokenizer loaded successfully.")
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompts,
    batch_size=2,
    max_seq_length=2048,
    output_dir="output/extractions/default_run",
    layers_to_extract=(0,5,10,15,20,25),
    attn_layers_to_extract=(0,5,10,15,20,25),
    top_k_logits=10,
    log_file="logs/inference_run.log",
    skip_save=False,
):
    """
    Reuses the *already-loaded* model & tokenizer to run inference on a 
    list of string 'prompts'. 
    Saves the extracted hidden states + final predictions in `output_dir` 
    unless skip_save=True.
    """

    # Possibly set up logging
    logger = logging.getLogger("MyInferenceRunner")
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info("=== Starting run_inference ===")
    logger.info(f"Number of prompts = {len(prompts)}")
    logger.info(f"output_dir = {output_dir}, skip_save={skip_save}")

    os.makedirs(output_dir, exist_ok=True)

    # Batching
    total_prompts = len(prompts)

    # Helper function
    def capture_activations(text_batch, idx):
        try:
            encodings = tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt"
            )
            input_ids = encodings["input_ids"].cuda()
            attention_mask = encodings["attention_mask"].cuda()

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True
                )
                generated_output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Extract the layers
            selected_hidden_states = {}
            for layer_idx in layers_to_extract:
                if layer_idx < len(outputs.hidden_states):
                    layer_tensor = outputs.hidden_states[layer_idx].cpu().to(torch.bfloat16)
                    selected_hidden_states[f"layer_{layer_idx}"] = layer_tensor

            selected_attentions = {}
            for layer_idx in attn_layers_to_extract:
                if layer_idx < len(outputs.attentions):
                    attn_tensor = outputs.attentions[layer_idx].cpu().to(torch.bfloat16)
                    selected_attentions[f"layer_{layer_idx}"] = attn_tensor

            # top-k logits
            logits = outputs.logits
            topk_vals, topk_indices = torch.topk(logits, k=top_k_logits, dim=-1)
            topk_vals = topk_vals.cpu().to(torch.bfloat16)
            topk_indices = topk_indices.cpu()

            final_predictions = [
                tokenizer.decode(pred, skip_special_tokens=True)
                for pred in generated_output.cpu()
            ]

            return {
                "hidden_states": selected_hidden_states,
                "attentions": selected_attentions,
                "topk_logits": topk_vals,
                "topk_indices": topk_indices,
                "input_ids": input_ids.cpu(),
                "final_predictions": final_predictions
            }
        except Exception as e:
            logger.exception(f"Error at batch {idx}: {e}")
            return None

    # Actually run
    for start_idx in range(0, total_prompts, batch_size):
        end_idx = start_idx + batch_size
        batch_texts = prompts[start_idx:end_idx]
        result = capture_activations(batch_texts, start_idx)
        if result is not None and not skip_save:
            # Save as .pt
            filename = os.path.join(output_dir, f"activations_{start_idx:05d}_{end_idx:05d}.pt")
            torch.save(result, filename)
            logger.info(f"Saved {filename}")

    logger.info("=== run_inference complete ===")
