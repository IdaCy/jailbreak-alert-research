# functions/csv_inference.py
import os
import math
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name="google/gemma-2-9b-it",
    use_bfloat16=True,
    hf_token=None,
    max_seq_length=2048
):
    """
    Loads the tokenizer and model into memory (once). Returns both objects.

    :param model_name: Name of the HF model repo.
    :param use_bfloat16: Whether to load model weights in bfloat16.
    :param hf_token: Optional HF auth token (string).
    :param max_seq_length: Just to store if needed. (Not always required here.)
    :return: (tokenizer, model) both on GPU, in eval mode.
    """
    # Clear GPU cache
    logger.info("Clearing CUDA cache")
    torch.cuda.empty_cache()

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU is available! Check your runtime or environment.")

    logger.info(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True, 
        use_auth_token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("No pad_token found, using eos_token as pad_token.")

    # Figure out GPU total memory to limit usage, if desired
    gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
    max_memory = {0: f"{int(gpu_mem * 0.9)}GB"}

    logger.info(f"Loading model {model_name} with device_map='auto', bfloat16={use_bfloat16}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
        use_auth_token=hf_token
    )
    model.eval()
    logger.info("Model loaded successfully. Returning tokenizer and model.")

    return tokenizer, model


def run_inference(
    model,
    tokenizer,
    prompts,
    output_dir="output/",
    batch_size=2,
    max_seq_length=2048,
    max_new_tokens=50,
    extract_hidden_layers=None,
    extract_attention_layers=None,
    top_k_logits=10,
    log_file="logs/inference.log",
    error_log="logs/errors.log"
):
    """
    Runs inference on a list of prompts and saves intermediate activations to .pt files.
    
    :param model: The loaded HuggingFace model (on GPU).
    :param tokenizer: The corresponding tokenizer.
    :param prompts: A list of strings (the prompts).
    :param output_dir: Directory where .pt files of activations should be saved.
    :param batch_size: How many prompts to process per forward pass.
    :param max_seq_length: Tokenization max length.
    :param max_new_tokens: How many new tokens to generate for each prompt.
    :param extract_hidden_layers: List of layer indices to store hidden_states from.
    :param extract_attention_layers: List of layer indices to store attentions from.
    :param top_k_logits: Store top-K logits from the forward pass (optional).
    :param log_file: Path to the main log file.
    :param error_log: Path to the error log file.
    :return: None (saves .pt files in the output directory).
    """
    if extract_hidden_layers is None:
        extract_hidden_layers = [0, 5, 10, 15]
    if extract_attention_layers is None:
        extract_attention_layers = [0, 5, 10, 15]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(os.path.dirname(error_log), exist_ok=True)

    logger.info("=== Starting run_inference ===")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Number of prompts: {len(prompts)}")
    logger.info(f"Batch size: {batch_size}, max_new_tokens: {max_new_tokens}")
    
    def capture_activations(text_batch, batch_idx):
        """Inner function to process a batch of texts and gather partial outputs."""
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

                # Generate text continuation
                generated = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Prepare partial outputs
            selected_hidden_states = {}
            for layer_idx in extract_hidden_layers:
                if layer_idx < len(outputs.hidden_states):
                    layer_tensor = outputs.hidden_states[layer_idx].cpu()
                    selected_hidden_states[f"layer_{layer_idx}"] = layer_tensor

            selected_attentions = {}
            for layer_idx in extract_attention_layers:
                if layer_idx < len(outputs.attentions):
                    attn_tensor = outputs.attentions[layer_idx].cpu()
                    selected_attentions[f"layer_{layer_idx}"] = attn_tensor

            logits = outputs.logits  # shape [batch, seq_len, vocab_size]
            topk_vals, topk_indices = torch.topk(logits, k=top_k_logits, dim=-1)

            # Convert to CPU
            topk_vals = topk_vals.cpu()
            topk_indices = topk_indices.cpu()

            final_predictions = [
                tokenizer.decode(pred, skip_special_tokens=True)
                for pred in generated.cpu()
            ]

            return {
                "hidden_states": selected_hidden_states,
                "attentions": selected_attentions,
                "topk_vals": topk_vals,
                "topk_indices": topk_indices,
                "input_ids": input_ids.cpu(),
                "final_predictions": final_predictions,
                "batch_texts": text_batch
            }

        except torch.cuda.OutOfMemoryError:
            logger.error(f"CUDA OOM at batch {batch_idx}. Clearing cache.")
            torch.cuda.empty_cache()
            with open(error_log, "a", encoding="utf-8") as err_log:
                err_log.write(f"OOM error at batch {batch_idx}.\n")
            return None

        except Exception as e:
            logger.exception(f"Error at batch {batch_idx}: {str(e)}")
            with open(error_log, "a", encoding="utf-8") as err_log:
                err_log.write(f"Error at batch {batch_idx}: {str(e)}\n")
            return None

    total_prompts = len(prompts)
    # Calculate how many total batches we'll have
    total_batches = math.ceil(total_prompts / batch_size)

    logger.info(f"Total batches to process: {total_batches}")

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_prompts)
        batch_texts = prompts[start_idx:end_idx]

        # Print/log every 20th batch
        if batch_idx % 20 == 0:
            logger.info(f"Processing batch {batch_idx} / {total_batches} "
                        f"(prompt indices {start_idx}-{end_idx})")
            print(f"== [Batch {batch_idx} of {total_batches}] "
                  f"Prompts {start_idx}-{end_idx} ==")  # direct print

        result = capture_activations(batch_texts, batch_idx)
        if result:
            save_path = os.path.join(output_dir, f"activations_{start_idx:05d}_{end_idx:05d}.pt")
            torch.save(result, save_path)
            logger.debug(f"Saved .pt activations => {save_path}")

    logger.info("=== run_inference complete ===")
    print("=== Inference is complete! ===")
