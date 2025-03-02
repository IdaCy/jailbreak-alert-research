import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------------------
# 1. Configuration and Setup
# ------------------------------------------------------------------------
PROMPT_FILE = globals().get("PROMPT_FILE", "data/ReNeLLM/jailbreaks/jb400.csv")
OUTPUT_DIR = globals().get("OUTPUT_DIR", "output/extractions/jailbreak")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = globals().get("MODEL_NAME", "google/gemma-2-2b")

BATCH_SIZE = globals().get("BATCH_SIZE", 2)
USE_BFLOAT16 = globals().get("USE_BFLOAT16", True)
MAX_SEQ_LENGTH = globals().get("MAX_SEQ_LENGTH", 2048)

# Store only these layers (e.g. every 5th)
EXTRACT_HIDDEN_LAYERS = globals().get("EXTRACT_HIDDEN_LAYERS", [0, 5, 10, 15, 20, 25])
EXTRACT_ATTENTION_LAYERS = globals().get("EXTRACT_ATTENTION_LAYERS", [0, 5, 10, 15, 20, 25])

# How many logits to keep
TOP_K_LOGITS = globals().get("TOP_K_LOGITS", 10)

LOG_FILE = globals().get("LOG_FILE", "logs/jb_run_progress.log")
ERROR_LOG = globals().get("ERROR_LOG", "logs/jb_run_errors.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(ERROR_LOG), exist_ok=True)

# Limit how many total samples to process
NUM_SAMPLES = globals().get("NUM_SAMPLES", None)
if isinstance(NUM_SAMPLES, str) and NUM_SAMPLES.isdigit():
    NUM_SAMPLES = int(NUM_SAMPLES)

# Hugging Face token from environment
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN:
    print(f"Using HF_TOKEN from environment: {HF_TOKEN[:8]}... (start bit)")
else:
    HF_TOKEN = globals().get("HF_TOKEN", None)
    if HF_TOKEN:
        print(f"Using HF_TOKEN from globals: {HF_TOKEN[:8]}... (start bit)")
    else:
        print("No HF_TOKEN found in environment or globals. Attempting to proceed without auth token.")

# ------------------------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------------------------
def load_sentences(file_path):
    """Reads a file line-by-line to ensure no unwanted splitting occurs."""
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]
    return pd.DataFrame(sentences, columns=["sentence"])

df_clean = load_sentences(PROMPT_FILE)
all_texts = df_clean['sentence'].tolist()

# If NUM_SAMPLES is specified, truncate the list
if NUM_SAMPLES is not None and NUM_SAMPLES < len(all_texts):
    all_texts = all_texts[:NUM_SAMPLES]

print(f"Loaded {len(all_texts)} samples for inference.")

# ------------------------------------------------------------------------
# 3. Managing GPU Memory
# ------------------------------------------------------------------------
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
    max_memory = {0: f"{int(gpu_mem * 0.9)}GB"}  # Use ~90% of GPU memory
else:
    raise RuntimeError("No GPUs available! Ensure you are running on a GPU node.")

# ------------------------------------------------------------------------
# 4. Load Model and Tokenizer
# ------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_auth_token=HF_TOKEN  # Pass the token if available
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # ensure tokenizer has a pad token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto",
    max_memory=max_memory,
    use_auth_token=HF_TOKEN  # same token
)
model.eval()

print("Model loaded successfully.")

# ------------------------------------------------------------------------
# 5. Function to Run Inference and Capture Activations
# ------------------------------------------------------------------------
def capture_activations(text_batch, idx):
    """Runs model inference on a batch and extracts limited layers + top-k logits."""
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

            # Also generate a short completion
            generated_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 1) Hidden states: only store selected layers
        selected_hidden_states = {}
        for layer_idx in EXTRACT_HIDDEN_LAYERS:
            if layer_idx < len(outputs.hidden_states):
                layer_tensor = outputs.hidden_states[layer_idx].cpu().to(torch.bfloat16)
                selected_hidden_states[f"layer_{layer_idx}"] = layer_tensor

        # 2) Attention: only store selected layers
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

        # 5) Return packaged data
        return {
            "hidden_states": selected_hidden_states,
            "attentions": selected_attentions,
            "topk_logits": topk_vals,
            "topk_indices": topk_indices,
            "input_ids": input_ids.cpu(),
            "final_predictions": final_predictions
        }

    except torch.cuda.OutOfMemoryError:
        with open(ERROR_LOG, "a") as err_log:
            err_log.write(f"OOM Error at index {idx}\n")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        with open(ERROR_LOG, "a") as err_log:
            err_log.write(f"Error at index {idx}: {str(e)}\n")
        return None

# ------------------------------------------------------------------------
# 6. Run Batch Inference and Save Activations
# ------------------------------------------------------------------------
print("Starting inference...")

for start_idx in range(0, len(all_texts), BATCH_SIZE):
    end_idx = start_idx + BATCH_SIZE
    batch_texts = all_texts[start_idx:end_idx]

    if start_idx % 1000 == 0:
        print(f"Processing batch {start_idx}/{len(all_texts)}...")

    activations = capture_activations(batch_texts, start_idx)
    if activations:
        filename = os.path.join(OUTPUT_DIR, f"activations_{start_idx:05d}_{end_idx:05d}.pt")
        torch.save(activations, filename)

    if start_idx % 5000 == 0:
        print(f"Saved activations up to sample {start_idx}")

print(f"Inference complete. Activations are stored in '{OUTPUT_DIR}'.")
