# functions/csv_inference.py
import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(
    model_name,
    hf_token=None,
    use_bfloat16=True,
    max_memory=None,
    logger=None
):
    """
    Loads the tokenizer and model from a specified Hugging Face model repo.
    Returns (model, tokenizer).
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("No pad token found; using eos_token as pad token.")
        
    logger.info(f"Loading model from {model_name} (bfloat16={use_bfloat16}, device_map=auto)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
        use_auth_token=hf_token
    )
    model.eval()
    logger.info("Model loaded successfully.")
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompts,
    output_dir,
    batch_size=4,
    max_seq_length=2048,
    extract_hidden_layers=None,
    extract_attention_layers=None,
    top_k_logits=10,
    logger=None
):
    """
    Runs inference in batches on a list of prompts, captures certain hidden states,
    attention layers, top-k logits, then saves to .pt files in `output_dir`.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    if extract_hidden_layers is None:
        extract_hidden_layers = [0, 5, 10, 15, 20, 25]
    if extract_attention_layers is None:
        extract_attention_layers = [0, 5, 10, 15, 20, 25]

    os.makedirs(output_dir, exist_ok=True)
    logger.info("=== Starting inference process ===")
    logger.info(f"Saving results to {output_dir}")

    total_prompts = len(prompts)
    logger.info(f"Total prompts: {total_prompts}")

    # Make sure we're on GPU
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available for inference.")
    torch.cuda.empty_cache()

    def capture_activations(text_batch, idx):
        """Runs model forward pass on a batch, returns hidden states & attentions & logits."""
        logger.debug(f"Encoding batch {idx} (size={len(text_batch)}) with max_length={max_seq_length}")
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
                # Optional: generate some short completion
                generated_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # 1) Hidden states
            selected_hidden_states = {}
            for layer_idx in extract_hidden_layers:
                if layer_idx < len(outputs.hidden_states):
                    layer_tensor = outputs.hidden_states[layer_idx].cpu()
                    selected_hidden_states[f"layer_{layer_idx}"] = layer_tensor

            # 2) Attention
            selected_attentions = {}
            for layer_idx in extract_attention_layers:
                if layer_idx < len(outputs.attentions):
                    attn_tensor = outputs.attentions[layer_idx].cpu()
                    selected_attentions[f"layer_{layer_idx}"] = attn_tensor

            # 3) Top-k logits
            logits = outputs.logits
            topk_vals, topk_indices = torch.topk(logits, k=top_k_logits, dim=-1)
            topk_vals = topk_vals.cpu()
            topk_indices = topk_indices.cpu()

            # 4) Decode generated text
            final_predictions = [
                tokenizer.decode(pred, skip_special_tokens=True)
                for pred in generated_output.cpu()
            ]

            return {
                "hidden_states": selected_hidden_states,
                "attentions": selected_attentions,
                "topk_vals": topk_vals,
                "topk_indices": topk_indices,
                "final_predictions": final_predictions
            }

        except torch.cuda.OutOfMemoryError:
            logger.error(f"CUDA OOM Error at batch {idx}. Clearing cache.")
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            logger.exception(f"Unhandled error at batch {idx}: {str(e)}")
            return None

    # Now iterate over all prompts in batches
    for start_idx in range(0, total_prompts, batch_size):
        end_idx = start_idx + batch_size
        batch_texts = prompts[start_idx:end_idx]

        if start_idx % 1000 == 0:
            logger.info(f"Processing batch {start_idx} / {total_prompts}...")

        activations = capture_activations(batch_texts, start_idx)
        if activations is not None:
            filename = os.path.join(output_dir, f"activations_{start_idx:05d}_{end_idx:05d}.pt")
            torch.save(activations, filename)
            logger.debug(f"Saved activations to {filename}")

    logger.info(f"Inference complete. Activations are stored in '{output_dir}'.")
