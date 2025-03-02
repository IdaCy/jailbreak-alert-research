import os
import torch
import pandas as pd
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------

NEUTRAL_FILE = globals().get("NEUTRAL_FILE", "/workspace/prompts/preprocessed/neutralQs.csv")
JAILBREAK_FILE = globals().get("JAILBREAK_FILE", "/workspace/prompts/preprocessed/jbQs.csv")
OUTPUT_DIR = globals().get("OUTPUT_DIR", "/workspace/gemma/extractions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = globals().get("MODEL_NAME", "google/gemma-2-2b")
HF_TOKEN = globals().get("HF_TOKEN", None)
if HF_TOKEN is None:
    raise ValueError("No Hugging Face token found in environment variable HF_TOKEN.")

BATCH_SIZE = globals().get("BATCH_SIZE", 16)
USE_BFLOAT16 = globals().get("USE_BFLOAT16", True)
MAX_SEQ_LENGTH = globals().get("MAX_SEQ_LENGTH", 512)
TOP_K_LOGITS = globals().get("TOP_K_LOGITS", 10)

EXTRACT_HIDDEN_LAYERS = globals().get("EXTRACT_HIDDEN_LAYERS", [0, 1, 2, 3, 4, 5, 10, 15, 20, 25])
EXTRACT_ATTENTION_LAYERS = globals().get("EXTRACT_ATTENTION_LAYERS", [10, 15, 20, 25])
FINAL_LAYER = globals().get("FINAL_LAYER", 25)

# New optional variable:
NUM_SAMPLES = globals().get("NUM_SAMPLES", None)  # If defined, only process this many samples

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

# 4) Load the model from local gemma-2-2b checkpoint
#    Remove device_map="auto" and then move the entire model to CUDA.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32,
    low_cpu_mem_usage=True,
    # device_map="auto",  <-- Remove this
    attn_implementation="eager",
    use_auth_token=HF_TOKEN
)

# If you added a new pad token, resize embeddings
model.resize_token_embeddings(len(tokenizer))

# Move entire model to GPU to match your input_ids.to("cuda")
model.to("cuda")

model.eval()
print("Model loaded successfully from local gemma2-2b checkpoint.")

# ------------------------------------------------------------------------
# Load Sentences
# ------------------------------------------------------------------------
def load_sentences(file_path):
    """Reads a file line-by-line and returns a list of sentences."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

neutral_texts = load_sentences(NEUTRAL_FILE)
jb_texts = load_sentences(JAILBREAK_FILE)

if len(neutral_texts) != len(jb_texts):
    raise ValueError("Mismatch between number of neutral and jb prompts!")

# If NUM_SAMPLES is set, truncate both lists
if NUM_SAMPLES is not None:
    neutral_texts = neutral_texts[:NUM_SAMPLES]
    jb_texts = jb_texts[:NUM_SAMPLES]

print(f"Loaded {len(neutral_texts)} samples for inference.")

# ------------------------------------------------------------------------
# Identify Relevant Token Indices
# ------------------------------------------------------------------------
def get_relevant_token_indices_pair(neutral_text, jb_text, tokenizer, window=3):
    """Find token positions that differ between neutral vs. jb versions."""
    tokens_neutral = tokenizer.tokenize(neutral_text)
    tokens_jb = tokenizer.tokenize(jb_text)
    sm = difflib.SequenceMatcher(None, tokens_neutral, tokens_jb)

    diff_indices_neutral, diff_indices_jb = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal':
            diff_indices_neutral.extend(range(i1, i2))
            diff_indices_jb.extend(range(j1, j2))

    def expand_indices(indices, max_len):
        expanded = set()
        for idx in indices:
            start = max(0, idx - window)
            end = min(max_len, idx + window + 1)
            expanded.update(range(start, end))
        return sorted(expanded)

    return (
        expand_indices(diff_indices_neutral, len(tokens_neutral)),
        tokens_neutral,
        expand_indices(diff_indices_jb, len(tokens_jb)),
        tokens_jb,
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
print("Starting inference & extraction of relevant activations for neutral & jb prompts...")

for start_idx in range(0, len(neutral_texts), BATCH_SIZE):
    end_idx = start_idx + BATCH_SIZE
    batch_neutral = neutral_texts[start_idx:end_idx]
    batch_jb = jb_texts[start_idx:end_idx]

    indices_neutral_batch, indices_jb_batch = [], []
    
    # Identify relevant tokens for each prompt
    for neutral_txt, jb_txt in zip(batch_neutral, batch_jb):
        rel_neutral, tokens_neutral, rel_jb, tokens_jb = get_relevant_token_indices_pair(neutral_txt, jb_txt, tokenizer)
        indices_neutral_batch.append(rel_neutral)
        indices_jb_batch.append(rel_jb)

    # Capture activations for the neutral version
    activations_neutral = capture_activations(batch_neutral, indices_neutral_batch)
    # Capture activations for the jb version
    activations_jb = capture_activations(batch_jb, indices_jb_batch)

    if activations_neutral and activations_jb:
        for i in range(len(batch_neutral)):
            sample_idx = start_idx + i
            # Save the final results for this sample
            filename = os.path.join(OUTPUT_DIR, f"activations_{sample_idx:05d}.pt")
            torch.save({"neutral": activations_neutral[i], "jb": activations_jb[i]}, filename)
        print(f"Saved activations for samples {start_idx} to {end_idx}")

print(f"Inference complete. Results saved in '{OUTPUT_DIR}'.")
