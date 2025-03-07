import os
import sys
import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

##############################################################################
# 1. Configuration
##############################################################################
# The CSV listing neurons to ablate. Must have "layer" and "neuron_idx" columns.
NEURONS_CSV = (
    os.environ.get("NEURONS_CSV")
    or globals().get("NEURONS_CSV", "analysis/gemma2b/good_jb_neuron/top_50_neurons_GLOBAL.csv")
)

# The prompts file to run inference on after ablation
PROMPT_FILE = (
    os.environ.get("PROMPT_FILE")
    or globals().get("PROMPT_FILE", "data/renellm/jb400.csv")
)

OUTPUT_DIR = (
    os.environ.get("OUTPUT_DIR")
    or globals().get("OUTPUT_DIR", "output/partial_neuron_ablation/gemma2b")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = (
    os.environ.get("MODEL_NAME")
    or globals().get("MODEL_NAME", "google/gemma-2-2b")
)

BATCH_SIZE = int(
    os.environ.get("BATCH_SIZE")
    or globals().get("BATCH_SIZE", 4)
)

USE_BFLOAT16 = (
    os.environ.get("USE_BFLOAT16")
    or globals().get("USE_BFLOAT16", True)
)
MAX_SEQ_LENGTH = int(
    os.environ.get("MAX_SEQ_LENGTH")
    or globals().get("MAX_SEQ_LENGTH", 2048)
)

LOG_FILE = (
    os.environ.get("LOG_FILE")
    or globals().get("LOG_FILE", "logs/partial_neuron_ablation.log")
)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

logger = logging.getLogger("PartialNeuronAblation")

logger.info("=== Starting partial neuron ablation script ===")
logger.info(f"NEURONS_CSV: {NEURONS_CSV}")
logger.info(f"PROMPT_FILE: {PROMPT_FILE}")
logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
logger.info(f"MODEL_NAME: {MODEL_NAME}")

##############################################################################
# 2. Load the CSV specifying (layer, neuron_idx) to ablate
##############################################################################
df_neurons = pd.read_csv(NEURONS_CSV)
# Must contain "layer" and "neuron_idx"
if "layer" not in df_neurons.columns or "neuron_idx" not in df_neurons.columns:
    raise ValueError("CSV must contain columns 'layer' and 'neuron_idx'!")

# Convert to int, just to be sure
df_neurons["layer"] = df_neurons["layer"].astype(int)
df_neurons["neuron_idx"] = df_neurons["neuron_idx"].astype(int)

# Group by layer => get the set of neurons for each layer
neurons_by_layer = {}
for row in df_neurons.itertuples():
    l = row.layer
    n = row.neuron_idx
    if l not in neurons_by_layer:
        neurons_by_layer[l] = []
    neurons_by_layer[l].append(n)

logger.info(f"Found ablation neurons for {len(neurons_by_layer)} distinct layers.")

##############################################################################
# 3. Load Prompt Data
##############################################################################
def load_prompts(file_path):
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                lines.append(line)
    return lines

prompts = load_prompts(PROMPT_FILE)
logger.info(f"Loaded {len(prompts)} prompts from {PROMPT_FILE}.")

##############################################################################
# 4. GPU Setup
##############################################################################
torch.cuda.empty_cache()
if not torch.cuda.is_available():
    raise RuntimeError("No GPU available.")

logger.info(f"Loading tokenizer for {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info(f"Loading model {MODEL_NAME} with bfloat16={USE_BFLOAT16}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32,
    device_map="auto",
)
model.eval()
logger.info("Model loaded successfully.")

##############################################################################
# 5. Define a forward hook to zero out certain neuron indices in MLP output
##############################################################################
# We'll attach this to each layer's .mlp submodule
def make_mlp_hook(layer_idx, neuron_indices_set):
    """
    neuron_indices_set: a Python set of neuron indices to zero out in the dimension [batch, seq_len, hidden_dim]
    We'll do out[..., idx] = 0.0 for idx in that set
    """
    def hook_fn(module, input, output):
        # output shape [batch, seq_len, hidden_dim]
        # we zero out the subset of neuron indices in hidden_dim dimension
        # If it’s a single tensor, we do out[..., i] = 0. If it’s a tuple, adapt accordingly.
        if not isinstance(output, torch.Tensor):
            # Possibly a tuple
            # Mistral/Llama MLP normally returns a single tensor
            # But if you see a tuple, adapt similarly to the self_attn logic
            logger.warning(f"Layer {layer_idx} MLP output was not a single tensor?! Skipping.")
            return output

        out_clone = output.clone()
        for idx in neuron_indices_set:
            if idx >= 0 and idx < out_clone.shape[-1]:
                out_clone[..., idx] = 0.0
        return out_clone
    return hook_fn

##############################################################################
# 6. Attach forward hooks for each layer with ablated neurons
##############################################################################
handles = []
for l_idx, n_list in neurons_by_layer.items():
    # If the layer index is out of range, skip
    if l_idx < 0 or l_idx >= len(model.model.layers):
        logger.warning(f"Specified layer {l_idx} is out of range, ignoring.")
        continue

    # We'll attach a single hook that zeros out all the relevant neurons in that layer
    neuron_set = set(n_list)  # for quick membership check
    mlp_module = getattr(model.model.layers[l_idx], "mlp", None)
    if mlp_module is None:
        logger.warning(f"Layer {l_idx} has no .mlp submodule? skipping.")
        continue

    h = mlp_module.register_forward_hook(make_mlp_hook(l_idx, neuron_set))
    handles.append(h)
    logger.info(f"Attached MLP forward hook for layer {l_idx} => zero {len(neuron_set)} neurons: {n_list}")

##############################################################################
# 7. Now run inference with partial neuron ablation
##############################################################################
def run_inference(text_batch, idx):
    try:
        enc = tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].cuda()
        attn_mask = enc["attention_mask"].cuda()

        with torch.no_grad():
            generated_output = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_output]
        return decoded
    except torch.cuda.OutOfMemoryError:
        logger.error(f"OOM at batch {idx}; clearing cache.")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        logger.exception(f"Error at batch {idx}: {e}")
        return None

logger.info("=== Starting partial neuron ablation inference ===")

all_results = []
n_prompts = len(prompts)
for start_idx in range(0, n_prompts, BATCH_SIZE):
    end_idx = start_idx + BATCH_SIZE
    batch_texts = prompts[start_idx:end_idx]

    preds = run_inference(batch_texts, start_idx)
    if preds is not None:
        for inp, outp in zip(batch_texts, preds):
            all_results.append({
                "prompt": inp,
                "response": outp
            })

ablated_csv = os.path.join(OUTPUT_DIR, "partial_neuron_ablation_results.csv")
pd.DataFrame(all_results).to_csv(ablated_csv, index=False, encoding="utf-8")
logger.info(f"Wrote partial neuron ablation results => {ablated_csv}")

##############################################################################
# 8. Cleanup
##############################################################################
logger.info("Removing forward hooks.")
for h in handles:
    h.remove()

logger.info("=== Done partial neuron ablation. ===")
