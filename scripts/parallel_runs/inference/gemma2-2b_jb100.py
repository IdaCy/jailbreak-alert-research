import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------------------
# 1. Configuration and Setup
# ------------------------------------------------------------------------

PROMPT_FILE = "data/ReNeLLM/jailbreaks/jb100.csv"
OUTPUT_DIR = "outputs/jailbreak"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_NAME = "google/gemma-2b"

BATCH_SIZE = 2
USE_BFLOAT16 = True  # Gemma-2B supports bfloat16, NOT float16!
MAX_SEQ_LENGTH = 2048  # Gemma supports longer sequences

LOG_FILE = "logs/jb_run_progress.log"
ERROR_LOG = "logs/jb_run_errors.log"

# ------------------------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------------------------

def load_sentences(file_path):
    """ Reads a file line-by-line to ensure no unwanted splitting occurs. """
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]
    return pd.DataFrame(sentences, columns=["sentence"])

df_clean = load_sentences(PROMPT_FILE)
all_texts = df_clean['sentence'].tolist()

print(f"Loaded {len(all_texts)} samples for inference.")

# ------------------------------------------------------------------------
# 3. Managing GPU Memory
# ------------------------------------------------------------------------

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Detect available GPUs and set memory constraints adaptively
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
    max_memory = {0: f"{int(gpu_mem * 0.9)}GB"}  # Use 90% of GPU memory
else:
    raise RuntimeError("No GPUs available! Ensure you are running on a GPU node.")

# ------------------------------------------------------------------------
# 4. Load Model and Tokenizer (Using Hugging Face's Gemma-2B)
# ------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Ensure tokenizer has a pad token

# Set up model with bfloat16 (not float16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto",  # Automatically selects available GPUs or CPU
    max_memory=max_memory
)
model.eval()

print("Model loaded successfully.")

# ------------------------------------------------------------------------
# 5. Function to Run Inference and Capture Activations
# ------------------------------------------------------------------------

def capture_activations(text_batch, idx):
    """ Runs model inference on a batch and extracts required activations. """
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            generated_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        hidden_states = outputs.hidden_states
        all_hidden_states = [layer.cpu().numpy() for layer in hidden_states]  # Convert to NumPy before saving

        # Decode generated text properly
        final_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in generated_output.cpu()]

        return {
            "hidden_states": all_hidden_states,
            "input_ids": input_ids.cpu(),
            "final_predictions": final_predictions  # Now properly decoded
        }

    except torch.cuda.OutOfMemoryError as e:
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
