#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

# ------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------

### START convenience
### only change those two variables and everything sets itself!
NAMEPREP_MODEL = (
    os.environ.get("NAMEPREP_MODEL")
    or globals().get("NAMEPREP_MODEL")
    or "gemma-3-4b-it"
).lower()
NAMEPREP_KEY = (
    os.environ.get("NAMEPREP_KEY")
    or globals().get("NAMEPREP_KEY")
    or "attack"
).lower()
### END convenience

NAMEPREP_OPTIONAL = ( # optional name addition for differentiating parallel runs
    os.environ.get("NAMEPREP_OPTIONAL")
    or globals().get("NAMEPREP_OPTIONAL")
    or ""
).lower()

DATA_SOURCE = (
    os.environ.get("DATA_SOURCE")
    or globals().get("DATA_SOURCE")
    or "json"      # "json" or "csv"
).lower()

PROMPT_FILE = (
    os.environ.get("PROMPT_FILE")
    or globals().get("PROMPT_FILE")
    or "data/renellm/attacks_all.json"
)

PROMPT_KEY = (  # relevant if DATA_SOURCE="json"
    os.environ.get("PROMPT_KEY")
    or globals().get("PROMPT_KEY", NAMEPREP_KEY)
)

OUTPUT_DIR = (
    os.environ.get("OUTPUT_DIR")
    or globals().get("OUTPUT_DIR")
    or "output/extractions/" + NAMEPREP_MODEL + "/" + NAMEPREP_OPTIONAL + NAMEPREP_KEY
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_REPO = (
    os.environ.get("MODEL_NAME")
    or globals().get("MODEL_NAME")
    or "google/" + NAMEPREP_MODEL
)

BATCH_SIZE = int(
    os.environ.get("BATCH_SIZE")
    or globals().get("BATCH_SIZE", 4)
)

"""USE_BFLOAT16 = (
    os.environ.get("USE_BFLOAT16")
    or globals().get("USE_BFLOAT16", True)
)
if isinstance(USE_BFLOAT16, str):
    USE_BFLOAT16 = (USE_BFLOAT16.lower() == "true")"""

MAX_SEQ_LENGTH = int(
    os.environ.get("MAX_SEQ_LENGTH")
    or globals().get("MAX_SEQ_LENGTH", 2048)
)

EXTRACT_HIDDEN_LAYERS = (
    os.environ.get("EXTRACT_HIDDEN_LAYERS")
    or globals().get("EXTRACT_HIDDEN_LAYERS", [0, 5, 10, 15, 20, 25])
)
if isinstance(EXTRACT_HIDDEN_LAYERS, str):
    EXTRACT_HIDDEN_LAYERS = [int(x.strip()) for x in EXTRACT_HIDDEN_LAYERS.strip("[]").split(",")]

EXTRACT_ATTENTION_LAYERS = (
    os.environ.get("EXTRACT_ATTENTION_LAYERS")
    or globals().get("EXTRACT_ATTENTION_LAYERS", [0, 5, 10, 15, 20, 25])
)
if isinstance(EXTRACT_ATTENTION_LAYERS, str):
    EXTRACT_ATTENTION_LAYERS = [int(x.strip()) for x in EXTRACT_ATTENTION_LAYERS.strip("[]").split(",")]

TOP_K_LOGITS = int(
    os.environ.get("TOP_K_LOGITS")
    or globals().get("TOP_K_LOGITS", 10)
)

LOG_FILE = (
    os.environ.get("LOG_FILE")
    or globals().get("LOG_FILE")
    or "logs/" + NAMEPREP_OPTIONAL + NAMEPREP_MODEL + "_" + NAMEPREP_KEY + "_inference.log"
)
ERROR_LOG = (
    os.environ.get("ERROR_LOG")
    or globals().get("ERROR_LOG")
    or "logs/" + NAMEPREP_OPTIONAL + NAMEPREP_MODEL + "_" + NAMEPREP_KEY + "_inference_errors.log"
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

# ------------------------------------------------------------------------
# 1a. Logging
# ------------------------------------------------------------------------
logger = logging.getLogger("Gemma3InferenceLogger")
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

logger.info("=== Starting Gemma3 Inference w/ Attention ===")
logger.info(f"Data source: {DATA_SOURCE}")
logger.info(f"Prompt file: {PROMPT_FILE}")
logger.info(f"Prompt key: {PROMPT_KEY} (if JSON)")


# ------------------------------------------------------------------------
# 0. Custom Transformer Layer (Captures Attention)
# ------------------------------------------------------------------------
class MyTransformerEncoderLayer(nn.Module):
    """
    A custom TransformerEncoderLayer that returns both hidden states and attention weights.
    Mirrors PyTorch's nn.TransformerEncoderLayer but exposes 'attn_weights'.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
        device=None,
        dtype=None
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first

        # Multihead self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first,
            device=device, dtype=dtype
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)

        if isinstance(activation, str):
            if activation == "relu":
                self.activation_func = F.relu
            elif activation == "gelu":
                self.activation_func = F.gelu
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation_func = activation  # custom/callable

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
    ):
        """
        src: shape [S, N, E] if batch_first=False, or [N, S, E] if batch_first=True
        Returns:
          output: hidden states after this layer
          attn_weights: [N, n_head, S, S] after we reshape
        """
        # Pre/post-norm style
        if self.norm_first:
            # norm-first approach
            src0 = self.norm1(src)
            attn_output, attn_weights = self.self_attn(
                src0, src0, src0,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True
            )
            src = src + self.dropout1(attn_output)
            src = src + self._feed_forward_chunk(self.norm2(src))
        else:
            # PyTorch default
            attn_output, attn_weights = self.self_attn(
                src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True
            )
            src = src + self.dropout1(attn_output)
            src = self.norm1(src)

            ff_out = self._feed_forward_chunk(src)
            src = src + self.dropout2(ff_out)
            src = self.norm2(src)

        return src, attn_weights

    def _feed_forward_chunk(self, x):
        x = self.linear1(x)
        x = self.activation_func(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x

# ------------------------------------------------------------------------
# 0a. Gemma 3 Model with Attention
# ------------------------------------------------------------------------
class Gemma3AttentionModel(nn.Module):
    """
    A minimal Gemma 3-like model that uses MyTransformerEncoderLayer, so we can capture attention.
    """

    def __init__(self, config, vocab_size):
        super().__init__()

        text_config = config.get("text_config", {})
        hidden_size = text_config.get("hidden_size", 2560)
        num_hidden_layers = text_config.get("num_hidden_layers", 34)
        intermediate_size = text_config.get("intermediate_size", 10240)
        n_head = text_config.get("num_attention_heads", 16)

        self.embed = nn.Embedding(vocab_size, hidden_size)

        # Build custom layers
        self.layers = nn.ModuleList([
            MyTransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_head,
                dim_feedforward=intermediate_size,
                dropout=0.1,
                activation="gelu",
                batch_first=False
            ) for _ in range(num_hidden_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        """
        Returns:
          logits: [batch, seq_len, vocab_size]
          all_hidden_states: list of shape [batch, seq_len, hidden_size]
          all_attentions: list of shape [batch, n_head, seq_len, seq_len]
        """
        # input_ids => [batch, seq_len]
        x = self.embed(input_ids)          # => [batch, seq_len, hidden_size]
        x = x.transpose(0, 1)              # => [seq_len, batch, hidden_size]

        all_hidden_states = []
        all_attentions = []

        # Each layer
        for layer in self.layers:
            x, attn = layer(x)
            # attn shape => [batch*n_head, seq_len, seq_len]
            # We'll reshape to [batch, n_head, seq_len, seq_len]
            seq_len, b_times_heads, S2 = attn.shape  # Actually MHA can be [batch*n_head, seq_len, seq_len]
            # but we need to re-check shape logic. Typically it's [batch*n_head, seq_len, seq_len].
            # Let's do it the more direct way:
            bsize = input_ids.size(0)
            nhead = attn.size(0) // bsize if bsize > 0 else 1
            # We'll guess that attn => [bsize*nhead, seq_len, seq_len].
            # So reshape:
            attn = attn.view(bsize, nhead, x.size(0), x.size(0))

            all_hidden_states.append(x.transpose(0, 1).clone())  # => [batch, seq_len, hidden_size]
            all_attentions.append(attn.clone())

        x = x.transpose(0, 1)  # => [batch, seq_len, hidden_size]
        logits = self.lm_head(x)
        return logits, all_hidden_states, all_attentions

# ------------------------------------------------------------------------
# 0b. Model Download & Loading
# ------------------------------------------------------------------------
def download_gemma3_model(model_repo="google/gemma-3-4b-it"):
    """
    Downloads from Hugging Face, returns local path, weights, config, vocab_size.
    """
    model_path = snapshot_download(repo_id=model_repo)
    print(f"Downloaded model to: {model_path}")

    # Find safetensors
    model_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    if not model_files:
        raise FileNotFoundError("No .safetensors file found in model repo!")

    model_file = os.path.join(model_path, model_files[0])
    weights = load_file(model_file)

    # Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as cf:
        config = json.load(cf)

    # Attempt a vocab size
    vocab_size = config.get("vocab_size", config.get("eoi_token_index", 256000) + 1)

    return model_path, weights, config, vocab_size

def load_gemma3_with_attention(model_repo="google/gemma-3-4b-it"):
    """
    Creates the Gemma3AttentionModel, loads weights, returns (model, config, vocab_size).
    """
    _, weights, config, vocab_size = download_gemma3_model(model_repo)
    model = Gemma3AttentionModel(config, vocab_size)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model, config, vocab_size

# ------------------------------------------------------------------------
# 0c. Minimal generation function (argmax) ignoring real text
# ------------------------------------------------------------------------
def generate_text(prompt, model, config, vocab_size, max_new_tokens=50):
    """
    Argmax-based generation, ignoring real prompt text. For demonstration only.
    """
    bos_token_id = config.get("boi_token_index", 255999)
    eos_token_id = config.get("eoi_token_index", 256000)

    input_ids = torch.tensor([[bos_token_id]], device="cuda")

    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _ = model(input_ids)
            next_token_logits = logits[:, -1, :]  # shape => [1, vocab_size]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            if next_token_id == eos_token_id:
                break
            generated_tokens.append(next_token_id)
            next_input = torch.tensor([[next_token_id]], device="cuda")
            input_ids = torch.cat([input_ids, next_input], dim=1)

    return " ".join(str(tid) for tid in generated_tokens)

# ------------------------------------------------------------------------
# 2. Prompt Loading Functions
# ------------------------------------------------------------------------
def load_json_prompts(file_path, json_key):
    """
    JSON structure like:
    [
      { "key1": "...", "key2": "...", ... },
      { "key1": "...", "key2": "...", ... },
      ...
    ]
    We return the text from the specified json_key.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of objects in JSON, got type={type(data)}")

    texts = []
    for i, item in enumerate(data):
        val = item.get(json_key, "").strip()
        if val:
            texts.append(val)
        else:
            logger.debug(f"No '{json_key}' found at index {i}, skipping.")
    return texts

def load_csv_prompts(file_path):
    """
    CSV file, but we ignore commas entirely, just read line by line.
    Return each line as a single prompt.
    """
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts

def load_prompts(file_path, data_source, prompt_key):
    """
    Decide which loader to use based on data_source.
    """
    if data_source == "json":
        return load_json_prompts(file_path, prompt_key)
    elif data_source == "csv":
        return load_csv_prompts(file_path)
    else:
        raise ValueError(f"Unknown data_source={data_source}")

# Load the prompts
all_texts = load_prompts(PROMPT_FILE, DATA_SOURCE, PROMPT_KEY)
if NUM_SAMPLES is not None and NUM_SAMPLES < len(all_texts):
    all_texts = all_texts[:NUM_SAMPLES]
logger.info(f"Loaded {len(all_texts)} prompts from {DATA_SOURCE.upper()} file '{PROMPT_FILE}'.")

# ------------------------------------------------------------------------
# 3. GPU Memory
# ------------------------------------------------------------------------
logger.info("Clearing CUDA cache.")
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if not torch.cuda.is_available():
    raise RuntimeError("No GPU available!")
gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
max_memory = {0: f"{int(gpu_mem * 0.9)}GB"}
logger.info(f"GPU is available. Setting max_memory={max_memory}")

# ------------------------------------------------------------------------
# 4. Load Gemma3 Model
# ------------------------------------------------------------------------
logger.info(f"Loading Gemma3 from HF repo '{MODEL_REPO}' ...")
model, config, vocab_size = load_gemma3_with_attention(model_repo=MODEL_REPO)
model = model.to("cuda").eval()

"""if USE_BFLOAT16:
    logger.info("Casting model to bfloat16.")
    model = model.to(torch.bfloat16)"""

# ------------------------------------------------------------------------
# 5. Inference Function (Capture everything)
# ------------------------------------------------------------------------
def capture_activations(text_batch, batch_idx):
    """
    1) Toy "tokenization" => [BOS].
    2) Forward => (logits, hidden_states, attentions).
    3) Filter layers by EXTRACT_HIDDEN_LAYERS, EXTRACT_ATTENTION_LAYERS.
    4) top-k from last logits position.
    5) Generate text for each prompt.
    """
    logger.debug(f"Encoding batch {batch_idx} (size={len(text_batch)})")
    bos_token_id = config.get("boi_token_index", 255999)

    try:
        # Build input batch
        input_list = [[bos_token_id] for _ in text_batch]
        #input_ids = torch.tensor(input_list, device="cuda")
        input_ids = torch.tensor(input_list, device="cuda", dtype=torch.long)
        #if USE_BFLOAT16:
        #    input_ids = input_ids.to(torch.bfloat16)

        with torch.no_grad():
            logits, all_hs, all_attn = model(input_ids)

            # A quick generation for each prompt
            final_predictions = []
            for p in text_batch:
                pred_str = generate_text(p, model, config, vocab_size, max_new_tokens=50)
                final_predictions.append(pred_str)

        # all_hs => list of shape [batch, seq_len, hidden_size] for each layer
        # all_attn => list of shape [batch, n_head, seq_len, seq_len] for each layer
        # logits => [batch, seq_len, vocab_size]

        # Gather selected hidden states
        selected_hidden_states = {}
        selected_attentions = {}
        num_layers = len(all_hs)  # total layers

        for layer_idx in EXTRACT_HIDDEN_LAYERS:
            if layer_idx < num_layers:
                selected_hidden_states[f"layer_{layer_idx}"] = all_hs[layer_idx].cpu()

        for layer_idx in EXTRACT_ATTENTION_LAYERS:
            if layer_idx < num_layers:
                selected_attentions[f"layer_{layer_idx}"] = all_attn[layer_idx].cpu()

        # top-k from last position
        last_pos_logits = logits[:, -1, :]  # [batch, vocab_size]
        topk_vals, topk_indices = torch.topk(last_pos_logits, k=TOP_K_LOGITS, dim=-1)

        return {
            "hidden_states": selected_hidden_states,
            "attentions": selected_attentions,
            "topk_logits": topk_vals.cpu(),
            "topk_indices": topk_indices.cpu(),
            "input_ids": input_ids.cpu(),
            "final_predictions": final_predictions
        }

    except torch.cuda.OutOfMemoryError:
        logger.error(f"OOM at batch {batch_idx}. Clearing cache.")
        with open(ERROR_LOG, "a") as ef:
            ef.write(f"OOM at batch {batch_idx}\n")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        logger.exception(f"Error at batch {batch_idx}: {str(e)}")
        with open(ERROR_LOG, "a") as ef:
            ef.write(f"Error at batch {batch_idx}: {str(e)}\n")
        return None

# ------------------------------------------------------------------------
# 6. Batched Inference & Saving
# ------------------------------------------------------------------------
logger.info("=== Begin inference ===")

total_prompts = len(all_texts)
for start_idx in range(0, total_prompts, BATCH_SIZE):
    end_idx = start_idx + BATCH_SIZE
    batch_texts = all_texts[start_idx:end_idx]

    if start_idx % 1000 == 0:
        logger.info(f"Processing batch {start_idx} / {total_prompts} ...")

    output_data = capture_activations(batch_texts, start_idx)
    if output_data is not None:
        outfile = os.path.join(OUTPUT_DIR, f"activations_{start_idx:05d}_{end_idx:05d}.pt")
        torch.save(output_data, outfile)
        logger.debug(f"Saved {outfile}")

    if start_idx % 5000 == 0 and start_idx > 0:
        logger.info(f"Saved up to batch {start_idx}")

logger.info(f"Inference complete! Files written to '{OUTPUT_DIR}'.")
