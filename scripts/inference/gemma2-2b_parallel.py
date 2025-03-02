import os
import torch
import pandas as pd
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------

CLEAN_FILE = globals().get("CLEAN_FILE", "/workspace/prompts/preprocessed/cleanQs.csv")
TYPO_FILE = globals().get("TYPO_FILE", "/workspace/prompts/preprocessed/typoQs.csv")
OUTPUT_DIR = globals().get("OUTPUT_DIR", "/workspace/gemma/extractions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = globals().get("MODEL_NAME", "google/gemma-2-2b")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN is None:
    raise ValueError("No Hugging Face token found in environment variable HF_TOKEN.")

BATCH_SIZE = globals().get("BATCH_SIZE", 16)
USE_BFLOAT16 = globals().get("USE_BFLOAT16", True)
MAX_SEQ_LENGTH = globals().get("MAX_SEQ_LENGTH", 512)
TOP_K_LOGITS = globals().get("TOP_K_LOGITS", 10)

EXTRACT_HIDDEN_LAYERS = globals().get("EXTRACT_HIDDEN_LAYERS", [0, 1, 2, 3, 4, 5, 10, 15, 20, 25])
EXTRACT_ATTENTION_LAYERS = globals().get("EXTRACT_ATTENTION_LAYERS", [10, 15, 20, 25])
FINAL_LAYER = globals().get("FINAL_LAYER", 25)

# ------------------------------------------------------------------------
# Load Model and Tokenizer
# ------------------------------------------------------------------------

# 1) Initialize tokenizer from local checkpoint
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_auth_token=HF_TOKEN,
    trust_remote_code=True
)

# 2) Ensure we have a proper pad token if none is defined
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'

# 3) Make sure pad_token_id is set in config or on the tokenizer
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
if getattr(tokenizer, "pad_token_id", None) is not None:
    config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
else:
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

# 4) Load the model from local gemma2-2b checkpoint
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="eager",
    use_auth_token=HF_TOKEN
)

# If you added a new pad token, resize embeddings
model.resize_token_embeddings(len(tokenizer))

model.eval()
print("Model loaded successfully from local gemma2-2b checkpoint.")

# ------------------------------------------------------------------------
# Load Sentences
# ------------------------------------------------------------------------
def load_sentences(file_path):
    """Reads a file line-by-line and returns a list of sentences."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

clean_texts = load_sentences(CLEAN_FILE)
typo_texts = load_sentences(TYPO_FILE)

if len(clean_texts) != len(typo_texts):
    raise ValueError("Mismatch between number of clean and typo prompts!")

print(f"Loaded {len(clean_texts)} samples for inference.")

# ------------------------------------------------------------------------
# Identify Relevant Token Indices
# ------------------------------------------------------------------------
def get_relevant_token_indices_pair(clean_text, typo_text, tokenizer, window=3):
    """Find token positions that differ between clean vs. typo versions."""
    tokens_clean = tokenizer.tokenize(clean_text)
    tokens_typo = tokenizer.tokenize(typo_text)
    sm = difflib.SequenceMatcher(None, tokens_clean, tokens_typo)

    diff_indices_clean, diff_indices_typo = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal':
            diff_indices_clean.extend(range(i1, i2))
            diff_indices_typo.extend(range(j1, j2))

    def expand_indices(indices, max_len):
        expanded = set()
        for idx in indices:
            start = max(0, idx - window)
            end = min(max_len, idx + window + 1)
            expanded.update(range(start, end))
        return sorted(expanded)

    return (
        expand_indices(diff_indices_clean, len(tokens_clean)),
        tokens_clean,
        expand_indices(diff_indices_typo, len(tokens_typo)),
        tokens_typo,
    )

# ------------------------------------------------------------------------
# Run Inference and Extract Features
# ------------------------------------------------------------------------
def capture_activations(text_batch, indices_batch):
    """Runs inference on a batch of texts, extracts relevant hidden states, attention, logits, etc."""
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
                output_attentions=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        batch_results = {}

        for i in range(len(text_batch)):
            relevant_indices = indices_batch[i] if indices_batch else list(range(len(input_ids[i])))

            sample_result = {}

            # Hidden states (only selected layers)
            sample_result["hidden_states"] = {
                f"layer_{l}": hidden_states[l][i][relevant_indices].to(torch.bfloat16).cpu()
                for l in EXTRACT_HIDDEN_LAYERS
            }

            # Attention scores (only selected attention layers)
            sample_result["attention_scores"] = {
                f"layer_{l}": attentions[l][i].mean(dim=0)[relevant_indices].to(torch.bfloat16).cpu()
                for l in EXTRACT_ATTENTION_LAYERS
            }

            # Final-layer top-k logits & probabilities
            top_logits, top_indices = torch.topk(logits[i][relevant_indices], k=TOP_K_LOGITS, dim=-1)
            top_probs = probabilities[i][relevant_indices].gather(-1, top_indices)

            sample_result["top_k_logits"] = {
                f"token_{t}": top_logits[j].to(torch.bfloat16).cpu()
                for j, t in enumerate(relevant_indices)
            }
            sample_result["top_k_probs"] = {
                f"token_{t}": top_probs[j].to(torch.bfloat16).cpu()
                for j, t in enumerate(relevant_indices)
            }
            sample_result["top_k_indices"] = {
                f"token_{t}": top_indices[j].to(torch.int16).cpu()
                for j, t in enumerate(relevant_indices)
            }

            # Generate a short completion for reference
            generated_ids = model.generate(
                input_ids=input_ids[i].unsqueeze(0),
                attention_mask=attention_mask[i].unsqueeze(0),
                max_new_tokens=20,
                do_sample=False
            )
            sample_result["predicted_text"] = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Store the tokens + original text
            sample_result["tokens"] = tokenizer.convert_ids_to_tokens(input_ids[i].cpu().tolist())
            sample_result["original_text"] = text_batch[i]

            batch_results[i] = sample_result

        return batch_results

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

# ------------------------------------------------------------------------
# Main Loop: Process All Prompts
# ------------------------------------------------------------------------
print("Starting inference & extraction of relevant activations for clean & typo prompts...")

for start_idx in range(0, len(clean_texts), BATCH_SIZE):
    end_idx = start_idx + BATCH_SIZE
    batch_clean, batch_typo = clean_texts[start_idx:end_idx], typo_texts[start_idx:end_idx]

    indices_clean_batch, indices_typo_batch = [], []
    
    # Identify relevant tokens for each prompt
    for clean_txt, typo_txt in zip(batch_clean, batch_typo):
        rel_clean, tokens_clean, rel_typo, tokens_typo = get_relevant_token_indices_pair(clean_txt, typo_txt, tokenizer)
        indices_clean_batch.append(rel_clean)
        indices_typo_batch.append(rel_typo)

    # Capture activations for the clean version
    activations_clean = capture_activations(batch_clean, indices_clean_batch)
    # Capture activations for the typo version
    activations_typo = capture_activations(batch_typo, indices_typo_batch)

    if activations_clean and activations_typo:
        for i in range(len(batch_clean)):
            sample_idx = start_idx + i
            # Save the final results for this sample
            filename = os.path.join(OUTPUT_DIR, f"activations_{sample_idx:05d}.pt")
            torch.save({"clean": activations_clean[i], "typo": activations_typo[i]}, filename)
        print(f"Saved activations for samples {start_idx} to {end_idx}")

print(f"Inference complete. Results saved in '{OUTPUT_DIR}'.")
