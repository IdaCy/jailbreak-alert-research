import os
import torch
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------------------
# 1. Configuration and Setup
# ------------------------------------------------------------------------
PROMPT_FILE = (
    os.environ.get("PROMPT_FILE")
    or globals().get("PROMPT_FILE")
    or "data/code_chameleon/formatted_prompts.csv"
)

OUTPUT_DIR = (
    os.environ.get("OUTPUT_DIR")
    or globals().get("OUTPUT_DIR")
    or "output/extractions/code_chameleon/gemma-2-2b"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = (
    os.environ.get("MODEL_NAME")
    or globals().get("MODEL_NAME")
    #or "google/gemma-2-2b"
    or "mistralai/Mistral-7B-v0.1"
)

BATCH_SIZE = int(
    os.environ.get("BATCH_SIZE")
    or globals().get("BATCH_SIZE", 2)
)

USE_BFLOAT16 = (
    os.environ.get("USE_BFLOAT16") 
    or globals().get("USE_BFLOAT16", True)
)
# If USE_BFLOAT16 might be a string from env, you can do:
# USE_BFLOAT16 = (os.environ.get("USE_BFLOAT16") or globals().get("USE_BFLOAT16", True))
# USE_BFLOAT16 = (True if str(USE_BFLOAT16).lower() == "true" else False)

MAX_SEQ_LENGTH = int(
    os.environ.get("MAX_SEQ_LENGTH")
    or globals().get("MAX_SEQ_LENGTH", 2048)
)

EXTRACT_HIDDEN_LAYERS = (
    os.environ.get("EXTRACT_HIDDEN_LAYERS")
    or globals().get("EXTRACT_HIDDEN_LAYERS", [0, 5, 10, 15, 20, 25, 30, 31])
)
EXTRACT_ATTENTION_LAYERS = (
    os.environ.get("EXTRACT_ATTENTION_LAYERS")
    or globals().get("EXTRACT_ATTENTION_LAYERS", [0, 10, 20, 30])
)

TOP_K_LOGITS = int(
    os.environ.get("TOP_K_LOGITS")
    or globals().get("TOP_K_LOGITS", 10)
)

LOG_FILE = (
    os.environ.get("LOG_FILE")
    or globals().get("LOG_FILE")
    or "logs/jb_small_run_progress.log"
)
ERROR_LOG = (
    os.environ.get("ERROR_LOG")
    or globals().get("ERROR_LOG")
    or "logs/jb_small_run_errors.log"
)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(ERROR_LOG), exist_ok=True)

# Limit how many total samples to process
NUM_SAMPLES = (
    os.environ.get("NUM_SAMPLES")
    or globals().get("NUM_SAMPLES", None)
)
if isinstance(NUM_SAMPLES, str) and NUM_SAMPLES.isdigit():
    NUM_SAMPLES = int(NUM_SAMPLES)

# Hugging Face token from environment or globals
HF_TOKEN = os.environ.get("HF_TOKEN", None)
hf_token_src = "environment" if HF_TOKEN else "none"
if not HF_TOKEN:
    # fallback to Python globals
    possible_token = globals().get("HF_TOKEN", None)
    if possible_token:
        HF_TOKEN = possible_token
        hf_token_src = "globals"

# ------------------------------------------------------------------------
# 1a. Set up Logging
# ------------------------------------------------------------------------
logger = logging.getLogger("ReNeLLMLogger")
logger.setLevel(logging.DEBUG)  # We capture everything from DEBUG up

# Create a file handler and set its level
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Create a console handler for immediate output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

logger.info("=== Starting inference script with comprehensive logging ===")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Error log: {ERROR_LOG}")
logger.info(f"Model name: {MODEL_NAME}")

if HF_TOKEN:
    logger.info(f"Using HF_TOKEN from {hf_token_src} (first 8 chars): {HF_TOKEN[:8]}...")
else:
    logger.warning("No HF_TOKEN found; proceeding without auth token")

# ------------------------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------------------------
def load_sentences(file_path):
    """Reads a CSV file using pandas to handle multi-line prompts."""
    logger.debug(f"Loading lines from {file_path}")
    df = pd.read_csv(file_path)
    return df

df_clean = load_sentences(PROMPT_FILE)
all_texts = df_clean['sentence'].tolist()

if NUM_SAMPLES is not None and NUM_SAMPLES < len(all_texts):
    logger.info(f"Truncating dataset to first {NUM_SAMPLES} lines.")
    all_texts = all_texts[:NUM_SAMPLES]

logger.info(f"Loaded {len(all_texts)} samples for inference from {PROMPT_FILE}.")

# ------------------------------------------------------------------------
# 3. Managing GPU Memory
# ------------------------------------------------------------------------
logger.info("Clearing CUDA cache and setting up GPU memory usage.")
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
    max_memory = {0: f"{int(gpu_mem * 0.9)}GB"}
    logger.info(f"GPU is available. Setting max_memory={max_memory}")
else:
    raise RuntimeError("No GPUs available! Ensure you are running on a GPU node.")

# ------------------------------------------------------------------------
# 4. Load Model and Tokenizer
# ------------------------------------------------------------------------
logger.info(f"Loading tokenizer from {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_auth_token=HF_TOKEN
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.debug("No pad token found on tokenizer; using eos_token as pad token.")

logger.info(f"Loading model from {MODEL_NAME} (bfloat16={USE_BFLOAT16}, device_map=auto)")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto",
    max_memory=max_memory,
    use_auth_token=HF_TOKEN
)
model.eval()
logger.info("Model loaded successfully.")

# ------------------------------------------------------------------------
# 5. Function to Run Inference and Capture Activations
# ------------------------------------------------------------------------
def capture_activations(text_batch, idx):
    """Runs model inference on a batch and extracts limited layers + top-k logits."""
    logger.debug(f"Encoding batch {idx} (size={len(text_batch)}) with max_length={MAX_SEQ_LENGTH}")
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

        logger.debug("Running forward pass to get hidden_states and attentions.")
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )

            logger.debug("Generating short completion from model.generate()")
            generated_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 1) Hidden states
        selected_hidden_states = {}
        for layer_idx in EXTRACT_HIDDEN_LAYERS:
            if layer_idx < len(outputs.hidden_states):
                layer_tensor = outputs.hidden_states[layer_idx].cpu().to(torch.bfloat16)
                selected_hidden_states[f"layer_{layer_idx}"] = layer_tensor

        # 2) Attention
        selected_attentions = {}
        for layer_idx in EXTRACT_ATTENTION_LAYERS:
            if layer_idx < len(outputs.attentions):
                attn_tensor = outputs.attentions[layer_idx].cpu().to(torch.bfloat16)
                selected_attentions[f"layer_{layer_idx}"] = attn_tensor

        # 3) Top-k logits
        logits = outputs.logits
        topk_vals, topk_indices = torch.topk(logits, k=TOP_K_LOGITS, dim=-1)
        topk_vals = topk_vals.cpu().to(torch.bfloat16)
        topk_indices = topk_indices.cpu()

        # 4) Decode generated text
        final_predictions = [
            tokenizer.decode(pred, skip_special_tokens=True)
            for pred in generated_output.cpu()
        ]

        logger.debug(f"Batch {idx} inference complete. Returning activations.")
        return {
            "hidden_states": selected_hidden_states,
            "attentions": selected_attentions,
            "topk_logits": topk_vals,
            "topk_indices": topk_indices,
            "input_ids": input_ids.cpu(),
            "final_predictions": final_predictions
        }

    except torch.cuda.OutOfMemoryError:
        logger.error(f"CUDA OOM Error at batch {idx}. Attempting to clear cache.")
        with open(ERROR_LOG, "a") as err_log:
            err_log.write(f"OOM Error at index {idx}\n")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        logger.exception(f"Unhandled error at batch {idx}: {str(e)}")
        with open(ERROR_LOG, "a") as err_log:
            err_log.write(f"Error at index {idx}: {str(e)}\n")
        return None

# ------------------------------------------------------------------------
# 6. Run Batch Inference and Save Activations
# ------------------------------------------------------------------------
logger.info("=== Starting inference process ===")

total_prompts = len(all_texts)
for start_idx in range(0, total_prompts, BATCH_SIZE):
    end_idx = start_idx + BATCH_SIZE
    batch_texts = all_texts[start_idx:end_idx]

    if start_idx % 1000 == 0:
        logger.info(f"Processing batch {start_idx} / {total_prompts}...")

    activations = capture_activations(batch_texts, start_idx)
    if activations:
        filename = os.path.join(OUTPUT_DIR, f"activations_{start_idx:05d}_{end_idx:05d}.pt")
        torch.save(activations, filename)
        logger.debug(f"Saved activations to {filename}")

    if start_idx % 5000 == 0 and start_idx > 0:
        logger.info(f"Saved activations up to batch {start_idx}")

logger.info(f"Inference complete. Activations are stored in '{OUTPUT_DIR}'.")
