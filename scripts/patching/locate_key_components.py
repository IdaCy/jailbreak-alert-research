import os
import sys
import logging
import math
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

###############################################################################
# 1. Configuration via Env Vars
###############################################################################
MODEL_NAME = (
    os.environ.get("MODEL_NAME")
    or globals().get("MODEL_NAME", "google/gemma-2-2b")
    #or globals().get("MODEL_NAME", "mistralai/Mistral-7B-v0.1")
)

PROMPT_FILE = (
    os.environ.get("PROMPT_FILE")
    or globals().get("PROMPT_FILE", "data/renellm/jb400.csv")
)

# Where we write the CSV with each component's importance
OUTPUT_DIR = (
    os.environ.get("OUTPUT_DIR")
    or globals().get("OUTPUT_DIR", "output/scan_key_components")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Multiple tokens allowed: AFFIRM_TOKENS="Sure,Yes", REFUSE_TOKENS="Sorry,Cannot"
AFFIRM_TOKENS = (
    os.environ.get("AFFIRM_TOKENS")
    or globals().get("AFFIRM_TOKENS", "Sure,Yes,Alright")
)
REFUSE_TOKENS = (
    os.environ.get("REFUSE_TOKENS")
    or globals().get("REFUSE_TOKENS", "Sorry,Cannot,No")
)

BATCH_SIZE = int(
    os.environ.get("BATCH_SIZE")
    or globals().get("BATCH_SIZE", 2)
)
USE_BFLOAT16 = (
    os.environ.get("USE_BFLOAT16")
    or globals().get("USE_BFLOAT16", True)
)
MAX_SEQ_LENGTH = int(
    os.environ.get("MAX_SEQ_LENGTH")
    or globals().get("MAX_SEQ_LENGTH", 512)
)

NUM_SAMPLES = (
    os.environ.get("NUM_SAMPLES")
    or globals().get("NUM_SAMPLES", None)
)
if isinstance(NUM_SAMPLES, str) and NUM_SAMPLES.isdigit():
    NUM_SAMPLES = int(NUM_SAMPLES)

# If HEAD_TYPE is "both", we'll do attention and MLP. If "attn_only", we skip MLP. If "mlp_only", we skip attention.
HEAD_TYPE = (
    os.environ.get("HEAD_TYPE")
    or globals().get("HEAD_TYPE", "both")
)
HEAD_TYPE = HEAD_TYPE.lower().strip()  # "both", "attn_only", or "mlp_only"

# HF token
HF_TOKEN = os.environ.get("HF_TOKEN", None)
hf_token_src = "env"
if not HF_TOKEN:
    HF_TOKEN = globals().get("HF_TOKEN", None)
    hf_token_src = "globals" if HF_TOKEN else "none"

###############################################################################
# 1a. Logging
###############################################################################
LOG_FILE = (
    os.environ.get("LOG_FILE")
    or globals().get("LOG_FILE", "logs/scan_components.log")
)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logger = logging.getLogger("ScanLogger")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(ch_formatter)
logger.addHandler(ch)

logger.info(f"=== Starting component-scanning script ===")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Prompts: {PROMPT_FILE}")
logger.info(f"Output Dir: {OUTPUT_DIR}")
logger.info(f"AFFIRM_TOKENS={AFFIRM_TOKENS}")
logger.info(f"REFUSE_TOKENS={REFUSE_TOKENS}")
logger.info(f"HEAD_TYPE={HEAD_TYPE}")

###############################################################################
# 2. Data Loading
###############################################################################
def load_prompts(file_path):
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines

prompts = load_prompts(PROMPT_FILE)
if NUM_SAMPLES is not None and NUM_SAMPLES < len(prompts):
    logger.info(f"Truncating to first {NUM_SAMPLES} prompts.")
    prompts = prompts[:NUM_SAMPLES]
logger.info(f"Loaded {len(prompts)} prompts total.")

###############################################################################
# 3. GPU / Device Setup
###############################################################################
torch.cuda.empty_cache()
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available, need a GPU.")
device_map = "auto"
gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
max_memory = {0: f"{int(gpu_mem * 0.9)}GB"}
logger.info(f"Using GPU with max_memory={max_memory}")

###############################################################################
# 4. Load Model & Tokenizer
###############################################################################
logger.info(f"Loading tokenizer for {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info(f"Loading model {MODEL_NAME} with bfloat16={USE_BFLOAT16}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32,
    low_cpu_mem_usage=True,
    device_map=device_map,
    max_memory=max_memory,
    use_auth_token=HF_TOKEN
)
model.eval()
logger.info("Model loaded successfully.")

###############################################################################
# 5. Handle multi-token sets for "affirm" vs "refuse"
#
# We'll parse them as comma-separated strings, then look up IDs.
# If an item is multiple subwords or is unknown, we skip it by default.
# Then for final logits:
#   score = sum( logits[refuse_ids] ) - sum( logits[affirm_ids] )
###############################################################################
def parse_token_list(token_list_str, tokenizer):
    """
    token_list_str = "Sure,Yes,Alright"
    We'll split by comma, strip each, tokenize.
    Only keep single-subword items. Return list of IDs.
    """
    items = [x.strip() for x in token_list_str.split(",")]
    valid_ids = []
    for it in items:
        # tokenize
        out = tokenizer(it, add_special_tokens=False)
        if len(out["input_ids"]) == 1:
            tid = out["input_ids"][0]
            valid_ids.append(tid)
        else:
            # multi-subword or empty => skip
            logger.debug(f"Skipping token '{it}' (not a single subword).")
    return list(set(valid_ids))  # deduplicate if repeated

affirm_id_list = parse_token_list(AFFIRM_TOKENS, tokenizer)
refuse_id_list = parse_token_list(REFUSE_TOKENS, tokenizer)

logger.info(f"Parsed AFFIRM tokens => {affirm_id_list} single subword IDs.")
logger.info(f"Parsed REFUSE tokens => {refuse_id_list} single subword IDs.")

if len(affirm_id_list) == 0 or len(refuse_id_list) == 0:
    logger.warning("No valid single-subword tokens found for affirm or refuse! "
                   "You might want to pick simpler tokens like 'Sure' or 'Yes' etc.")

def compute_logit_diff(final_logits):
    """
    final_logits: shape [vocab_size]
    We'll sum the logits for all refuse tokens, sum the logits for all affirm tokens,
    and return (sumRefuse - sumAffirm).
    If a token ID is out of range, we treat it as -inf (unlikely).
    """
    # final_logits is shape [vocab_size], on CPU or CUDA
    sum_refuse = 0.0
    for tid in refuse_id_list:
        if 0 <= tid < final_logits.shape[0]:
            sum_refuse += final_logits[tid].item()
        else:
            sum_refuse += float('-inf')

    sum_affirm = 0.0
    for tid in affirm_id_list:
        if 0 <= tid < final_logits.shape[0]:
            sum_affirm += final_logits[tid].item()
        else:
            sum_affirm += float('-inf')

    return sum_refuse - sum_affirm

###############################################################################
# 5b. Function to do 1 forward pass (optionally with ablation), measure final logit difference
###############################################################################
@torch.no_grad()
def get_logit_diff(prompt, ablate_hook=None):
    """
    ablate_hook is a function(layer_module, input, output) => zero out / partial mod, or None for no ablation.
    We'll do a single forward pass, get final token logits, then compute a multi-token difference.
    """
    inputs = tokenizer(
        prompt,
        padding=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()

    # If hooking is requested, register forward hooks
    handles = []
    if ablate_hook is not None:
        for l_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn") and ("attn" in HEAD_TYPE or "both" in HEAD_TYPE):
                h = layer.self_attn.register_forward_hook(ablate_hook)
                handles.append(h)
            if hasattr(layer, "mlp") and ("mlp" in HEAD_TYPE or "both" in HEAD_TYPE):
                h = layer.mlp.register_forward_hook(ablate_hook)
                handles.append(h)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    # remove hooks
    for h in handles:
        h.remove()

    # shape [batch_size, seq_len, vocab_size]
    logits = out.logits
    final_logits = logits[0, -1, :]  # last token distribution
    return compute_logit_diff(final_logits)

###############################################################################
# 6. Identify # of layers, # of heads for scanning
###############################################################################
num_layers = len(model.model.layers)
num_heads = model.model.layers[0].self_attn.num_heads  # assume consistent across layers

###############################################################################
# 7. Baseline: gather logit diffs with no ablation
###############################################################################
logger.info("Collecting baseline logit differences for each prompt.")
baseline_diffs = []
for prompt in prompts:
    diff = get_logit_diff(prompt, ablate_hook=None)
    baseline_diffs.append(diff)
avg_baseline_diff = sum(baseline_diffs) / len(baseline_diffs) if baseline_diffs else 0.0
logger.info(f"Average baseline logit diff over {len(prompts)} prompts: {avg_baseline_diff:.4f}")

results = []

###############################################################################
# 8. Build a small "hook factory" that zeroes out either 1 head or all MLP
###############################################################################
def head_ablation_factory(layer_idx, head_idx, num_heads, is_attn=True):
    """
    Return a forward hook that zeroes out the output for `head_idx` if is_attn=True,
    or zero entire MLP if is_attn=False.
    """
    hidden_dim = model.model.layers[layer_idx].self_attn.q_proj.weight.shape[0]
    head_dim = hidden_dim // num_heads

    def hook_fn(module, inp, out):
        if not is_attn:
            # MLP => zero entire out
            return torch.zeros_like(out)
        else:
            # out shape: [batch, seq_len, hidden_dim]
            # we want out[..., start:end] = 0
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            out[..., start:end] = 0.0
            return out

    return hook_fn

###############################################################################
# 9a. If HEAD_TYPE includes attention, measure each head's effect
###############################################################################
if HEAD_TYPE in ["both", "attn_only"]:
    logger.info("=== Scanning each attention head for importance. ===")
    for layer_idx in range(num_layers):
        for h_idx in range(num_heads):
            ab_fn = head_ablation_factory(layer_idx, h_idx, num_heads, is_attn=True)

            diffs_abl = []
            for prompt in prompts:
                d = get_logit_diff(prompt, ab_fn)
                diffs_abl.append(d)
            avg_abl = sum(diffs_abl)/len(diffs_abl)
            score = avg_abl - avg_baseline_diff

            results.append({
                "type": "attn",
                "layer": layer_idx,
                "head": h_idx,
                "mean_logit_diff_ablation": avg_abl,
                "importance_score": score
            })
    logger.info("Finished scanning attention heads.")

###############################################################################
# 9b. If HEAD_TYPE includes MLP, measure each layer's MLP effect
###############################################################################
if HEAD_TYPE in ["both", "mlp_only"]:
    logger.info("=== Scanning each MLP layer for importance. ===")
    for layer_idx in range(num_layers):
        ab_fn = head_ablation_factory(layer_idx, 0, 1, is_attn=False)

        diffs_abl = []
        for prompt in prompts:
            d = get_logit_diff(prompt, ab_fn)
            diffs_abl.append(d)
        avg_abl = sum(diffs_abl)/len(diffs_abl)
        score = avg_abl - avg_baseline_diff

        results.append({
            "type": "mlp",
            "layer": layer_idx,
            "head": None,
            "mean_logit_diff_ablation": avg_abl,
            "importance_score": score
        })
    logger.info("Finished scanning MLP layers.")

###############################################################################
# 10. Save results
###############################################################################
df = pd.DataFrame(results)
out_csv = os.path.join(OUTPUT_DIR, "component_importance.csv")
df.to_csv(out_csv, index=False)
logger.info(f"Wrote results to {out_csv}")
logger.info("=== Done ===")
