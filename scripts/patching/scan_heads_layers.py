import os
import torch
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

###############################################################################
# 1. Configuration and Setup
###############################################################################
PROMPT_FILE = (
    os.environ.get("PROMPT_FILE")
    or globals().get("PROMPT_FILE")
    or "data/renellm/jb400.csv"
)

OUTPUT_DIR = (
    os.environ.get("OUTPUT_DIR")
    or globals().get("OUTPUT_DIR")
    or "output/extractions/mistral7b/jb_small_ablation"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = (
    os.environ.get("MODEL_NAME")
    or globals().get("MODEL_NAME")
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
# USE_BFLOAT16 = (True if str(USE_BFLOAT16).lower() == "true" else False)

MAX_SEQ_LENGTH = int(
    os.environ.get("MAX_SEQ_LENGTH")
    or globals().get("MAX_SEQ_LENGTH", 2048)
)

TOP_K_LOGITS = int(
    os.environ.get("TOP_K_LOGITS")
    or globals().get("TOP_K_LOGITS", 10)
)

LOG_FILE = (
    os.environ.get("LOG_FILE")
    or globals().get("LOG_FILE")
    or "logs/jb_ablation_run_progress.log"
)
ERROR_LOG = (
    os.environ.get("ERROR_LOG")
    or globals().get("ERROR_LOG")
    or "logs/jb_ablation_run_errors.log"
)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(ERROR_LOG), exist_ok=True)

NUM_SAMPLES = (
    os.environ.get("NUM_SAMPLES")
    or globals().get("NUM_SAMPLES", None)
)
if isinstance(NUM_SAMPLES, str) and NUM_SAMPLES.isdigit():
    NUM_SAMPLES = int(NUM_SAMPLES)

HF_TOKEN = os.environ.get("HF_TOKEN", None)
hf_token_src = "environment" if HF_TOKEN else "none"
if not HF_TOKEN:
    possible_token = globals().get("HF_TOKEN", None)
    if possible_token:
        HF_TOKEN = possible_token
        hf_token_src = "globals"

# ----------------------------------------------------------------------
# New environment variables for patching
# Provide comma-separated lists of layer:head. E.g. "21:14,26:4"
# Or for MLP, just comma-separated layer indices "22,25,30"
# If you leave these blank, no ablation is done.
# ----------------------------------------------------------------------
HEADS_TO_ABLATE = (
    os.environ.get("HEADS_TO_ABLATE")  # e.g. "21:14,26:4"
    or globals().get("HEADS_TO_ABLATE", "")
)

MLPS_TO_ABLATE = (
    os.environ.get("MLPS_TO_ABLATE")   # e.g. "22,25"
    or globals().get("MLPS_TO_ABLATE", "")
)

# PATCH_METHOD can be "zero" or "randomize".
PATCH_METHOD = (
    os.environ.get("PATCH_METHOD")
    or globals().get("PATCH_METHOD", "zero")
).lower()  # "zero" or "randomize"

###############################################################################
# 1a. Set up Logging
###############################################################################
logger = logging.getLogger("AblationLogger")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

logger.info("=== Starting ablation inference script ===")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Error log: {ERROR_LOG}")
logger.info(f"Model name: {MODEL_NAME}")

if HF_TOKEN:
    logger.info(f"Using HF_TOKEN from {hf_token_src} (first 8 chars): {HF_TOKEN[:8]}...")
else:
    logger.warning("No HF_TOKEN found; proceeding without auth token")

###############################################################################
# 2. Load Data
###############################################################################
def load_sentences(file_path):
    logger.debug(f"Loading lines from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]
    return pd.DataFrame(sentences, columns=["sentence"])

df_clean = load_sentences(PROMPT_FILE)
all_texts = df_clean['sentence'].tolist()

if NUM_SAMPLES is not None and NUM_SAMPLES < len(all_texts):
    logger.info(f"Truncating dataset to first {NUM_SAMPLES} lines.")
    all_texts = all_texts[:NUM_SAMPLES]

logger.info(f"Loaded {len(all_texts)} samples from {PROMPT_FILE}.")

###############################################################################
# 3. GPU Setup
###############################################################################
logger.info("Clearing CUDA cache and setting up GPU memory usage.")
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
    max_memory = {0: f"{int(gpu_mem * 0.9)}GB"}
    logger.info(f"GPU is available. Setting max_memory={max_memory}")
else:
    raise RuntimeError("No GPUs available! Ensure you are running on a GPU node.")

###############################################################################
# 4. Load Model and Tokenizer
###############################################################################
logger.info(f"Loading tokenizer from {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_auth_token=HF_TOKEN
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

###############################################################################
# 5. Ablation / Patching Routines
###############################################################################
def parse_heads_to_ablate(head_string):
    """
    head_string like '21:14,26:4' => returns [(21,14), (26,4)]
    """
    if not head_string.strip():
        return []
    pairs = []
    for x in head_string.split(","):
        x = x.strip()
        if ":" in x:
            layer_str, head_str = x.split(":")
            if layer_str.isdigit() and head_str.isdigit():
                pairs.append((int(layer_str), int(head_str)))
    return pairs

def parse_mlps_to_ablate(mlp_string):
    """
    mlp_string like '22,25' => returns [22,25]
    """
    if not mlp_string.strip():
        return []
    layers = []
    for x in mlp_string.split(","):
        x = x.strip()
        if x.isdigit():
            layers.append(int(x))
    return layers

HEAD_PAIRS = parse_heads_to_ablate(HEADS_TO_ABLATE)
MLP_LAYERS = parse_mlps_to_ablate(MLPS_TO_ABLATE)

logger.info(f"Ablation: HEADS_TO_ABLATE={HEAD_PAIRS}, MLP_LAYERS_TO_ABLATE={MLP_LAYERS}, patch_method={PATCH_METHOD}")

def zero_or_random_(tensor, method="zero"):
    """
    Utility to zero out or randomize a tensor in-place.
    """
    with torch.no_grad():
        if method.lower() == "zero":
            tensor.zero_()
        elif method.lower() == "randomize":
            # e.g. normal(0,1) or something suitable
            torch.nn.init.normal_(tensor, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unsupported patch method: {method}")

def ablate_specific_attention_head(model, layer_idx, head_idx, method="zero"):
    """
    Zero or randomize the weights corresponding to a single attention head.
    We'll override the relevant slices of Wq, Wk, Wv, and Wo in that layer.
    WARNING: This is approximate—some architectures differ in shapes or naming.
    """
    try:
        # For Mistral or Llama-like HF architectures, you might find them under:
        # model.model.layers[layer_idx].self_attn...
        attn_module = model.model.layers[layer_idx].self_attn
        hidden_size = attn_module.q_proj.weight.shape[0]
        num_heads = attn_module.num_heads
        head_dim = hidden_size // num_heads

        # The slice for head_idx is [head_idx*head_dim : (head_idx+1)*head_dim]
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim

        # Q-proj, K-proj, V-proj, O-proj are linear layers. We'll zero the chunk of rows or columns
        # that correspond to the singled-out head. Typically, these are shaped [hidden_size, hidden_size].
        # The dimension with hidden_size is the input or output side. We want to remove the "output channel"
        # for that head in Q/K/V. Similarly in O-proj, we remove the "input channel" slice. This can vary.
        # We'll do the "out_features x in_features" approach for HF's default Linear.
        # So for Q-proj weight => shape is (hidden_size, hidden_size).
        # The 'output channel' dimension is dim=0. We'll ablate rows for Q/K/V. For O, we ablate columns.

        # (1) Q-proj
        zero_or_random_(attn_module.q_proj.weight.data[start:end, :], method)
        if attn_module.q_proj.bias is not None:
            zero_or_random_(attn_module.q_proj.bias.data[start:end], method)
        # (2) K-proj
        zero_or_random_(attn_module.k_proj.weight.data[start:end, :], method)
        if attn_module.k_proj.bias is not None:
            zero_or_random_(attn_module.k_proj.bias.data[start:end], method)
        # (3) V-proj
        zero_or_random_(attn_module.v_proj.weight.data[start:end, :], method)
        if attn_module.v_proj.bias is not None:
            zero_or_random_(attn_module.v_proj.bias.data[start:end], method)

        # (4) O-proj:
        # O-proj typically has shape [hidden_size, hidden_size]. The "input" dimension is dim=1.
        zero_or_random_(attn_module.o_proj.weight.data[:, start:end], method)
        # If bias is present, it's shape [hidden_size], so it affects the entire layer, usually we *don't* ablate that.
        # but to fully remove the head's effect, you *could* do some partial approach. We'll skip that here.

        logger.info(f"Ablated head {head_idx} in layer {layer_idx} with '{method}'.")
    except Exception as e:
        logger.warning(f"Failed to ablate head {head_idx} in layer {layer_idx}: {e}")

def ablate_specific_mlp_layer(model, layer_idx, method="zero"):
    """
    Zero or randomize the entire MLP block in the given layer.
    In many HF-based architectures, the MLP is typically feed_forward. We'll try:
      model.model.layers[layer_idx].mlp (Mistral) or .feed_forward or .mlp.up_proj, .mlp.down_proj, etc.
    """
    try:
        # For Mistral or Llama-based HF variants:
        mlp_module = model.model.layers[layer_idx].mlp
        # Typically has up_proj and down_proj (and maybe mid_proj, gating, etc.):
        zero_or_random_(mlp_module.up_proj.weight.data, method)
        zero_or_random_(mlp_module.up_proj.bias.data, method)
        zero_or_random_(mlp_module.down_proj.weight.data, method)
        zero_or_random_(mlp_module.down_proj.bias.data, method)
        # If there's a mid_proj or gate_proj, you can do likewise:
        if hasattr(mlp_module, "mid_proj"):
            zero_or_random_(mlp_module.mid_proj.weight.data, method)
            zero_or_random_(mlp_module.mid_proj.bias.data, method)

        logger.info(f"Ablated MLP layer {layer_idx} with '{method}'.")
    except Exception as e:
        logger.warning(f"Failed to ablate MLP layer {layer_idx}: {e}")

# Now run all the ablations we want:
for (l_idx, h_idx) in HEAD_PAIRS:
    ablate_specific_attention_head(model, l_idx, h_idx, method=PATCH_METHOD)

for l_idx in MLP_LAYERS:
    ablate_specific_mlp_layer(model, l_idx, method=PATCH_METHOD)

logger.info("All requested heads/MLPs have been ablated.")

###############################################################################
# 6. Inference (capturing minimal or no hidden states)
###############################################################################
def run_inference(text_batch, idx):
    """Runs model inference on a batch. 
       Here, we'll skip capturing all hidden states for brevity, 
       or you can adapt from your old script as needed.
    """
    logger.debug(f"Encoding batch {idx} (size={len(text_batch)})")
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
            generated_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        final_predictions = [
            tokenizer.decode(pred, skip_special_tokens=True)
            for pred in generated_output.cpu()
        ]
        return final_predictions

    except torch.cuda.OutOfMemoryError:
        logger.error(f"CUDA OOM Error at batch {idx}. Clearing cache.")
        with open(ERROR_LOG, "a") as err_log:
            err_log.write(f"OOM Error at index {idx}\n")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        logger.exception(f"Unhandled error at batch {idx}: {str(e)}")
        with open(ERROR_LOG, "a") as err_log:
            err_log.write(f"Error at index {idx}: {str(e)}\n")
        return None

###############################################################################
# 7. Run Batch Inference
###############################################################################
logger.info("=== Starting ablated inference process ===")

total_prompts = len(all_texts)
all_responses = []
for start_idx in range(0, total_prompts, BATCH_SIZE):
    end_idx = start_idx + BATCH_SIZE
    batch_texts = all_texts[start_idx:end_idx]

    if start_idx % 1000 == 0:
        logger.info(f"Processing batch {start_idx} / {total_prompts}...")

    preds = run_inference(batch_texts, start_idx)
    if preds is not None:
        for inp, outp in zip(batch_texts, preds):
            all_responses.append({
                "prompt": inp,
                "response": outp
            })

# Save all results to a CSV or JSON
out_csv = os.path.join(OUTPUT_DIR, "ablated_inference_results.csv")
pd.DataFrame(all_responses).to_csv(out_csv, index=False, encoding="utf-8")
logger.info(f"Ablated inference complete. Results saved to {out_csv}")
