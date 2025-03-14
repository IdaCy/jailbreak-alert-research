import os
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

# We'll import sentencepiece to parse .model if it exists.
try:
    import sentencepiece as spm
    _HAS_SPM = True
except ImportError:
    _HAS_SPM = False

def _infer_vocab_size(tokenizer_dir: str, config: dict) -> int:
    """
    Attempts to infer the vocab size from various tokenizer files:
    1. tokenizer_config.json
    2. tokenizer.json
    3. tokenizer.model (SentencePiece)
    If none are found or they don't contain vocab_size, fallback to eoi_token_index + 1.
    """
    eoi_token_idx = config.get("eoi_token_index", 256000)
    fallback_vocab_size = eoi_token_idx + 1

    # 1) tokenizer_config.json
    tcfg_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    if os.path.exists(tcfg_path):
        with open(tcfg_path, "r") as f:
            tcfg = json.load(f)
        if "vocab_size" in tcfg:
            return tcfg["vocab_size"]

    # 2) tokenizer.json
    tj_path = os.path.join(tokenizer_dir, "tokenizer.json")
    if os.path.exists(tj_path):
        with open(tj_path, "r") as f:
            tj = json.load(f)
        # Many tokenizers store the vocab in "model" -> "vocab"
        # or might store a separate "added_tokens" section.
        # We'll try "model" -> "vocab".
        try:
            possible_vocab = tj.get("model", {}).get("vocab", {})
            return len(possible_vocab)
        except Exception:
            pass

    # 3) tokenizer.model (SentencePiece)
    sp_path = os.path.join(tokenizer_dir, "tokenizer.model")
    if os.path.exists(sp_path) and _HAS_SPM:
        sp = spm.SentencePieceProcessor()
        sp.load(sp_path)
        return sp.vocab_size()

    # If everything fails, fallback to eoi_token_index + 1
    return fallback_vocab_size


def download_gemma3_model(model_repo="google/gemma-3-4b-it"):
    """
    Downloads the Gemma 3 model from Hugging Face and loads its weights and configuration.
    
    Returns:
        model_path (str): The path where the model files are stored.
        weights (dict): The model weights loaded from safetensors.
        config (dict): The model configuration.
        vocab_size (int): Vocabulary size extracted from tokenizer or fallback.
    """
    # Download model files
    model_path = snapshot_download(repo_id=model_repo)
    print(f"Model downloaded to: {model_path}")

    # Find safetensors model file
    model_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    if not model_files:
        raise FileNotFoundError("No safetensors model file found!")

    model_file = os.path.join(model_path, model_files[0])
    print(f"Loading model weights from: {model_file}")

    # Load weights
    weights = load_file(model_file)

    # Cast weights to FP16 to reduce memory usage
    for k in list(weights.keys()):
        weights[k] = weights[k].half()

    # Load config.json
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    print("Loaded Model Config:", config)

    # Infer vocab_size from tokenizer files or fallback
    vocab_size = _infer_vocab_size(model_path, config)
    print(f"Inferred vocab_size = {vocab_size}")

    return model_path, weights, config, vocab_size


class Gemma3Model(nn.Module):
    """
    A simplified transformer-based model for Gemma 3.
    """

    def __init__(self, config, vocab_size):
        super().__init__()

        # Extract hidden_size & num_layers
        text_config = config.get("text_config", {})
        hidden_size = text_config.get("hidden_size", 2560)
        num_hidden_layers = text_config.get("num_hidden_layers", 34)

        # Set up model components
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=16,  # Usually, nhead ~ sqrt(hidden_size)
                dim_feedforward=text_config.get("intermediate_size", 10240),
                activation="gelu"
            ) for _ in range(num_hidden_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        x = self.embed(input_ids)
        # nn.Transformer expects shape [seq_len, batch_size, d_model], so we transpose
        x = x.transpose(0, 1)  # shape => [seq_len, batch_size, hidden_size]
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1)  # back to [batch_size, seq_len, hidden_size]
        return self.lm_head(x)


def load_gemma3_model(model_repo="google/gemma-3-4b-it"):
    """
    Downloads and initializes the Gemma 3 model.
    
    Returns:
        model (Gemma3Model): The initialized model with loaded weights.
        config (dict): The model configuration.
        vocab_size (int): Vocabulary size.
    """
    model_path, weights, config, vocab_size = download_gemma3_model(model_repo)

    # Create model
    #model = Gemma3Model(config, vocab_size)
    model = Gemma3Model(config, vocab_size).half()

    # Load weights into model (strict=False to ignore mismatches)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval().to("cuda")

    return model, config, vocab_size


def generate_text(prompt, model, config, vocab_size, max_new_tokens=50):
    """
    Generates text using the Gemma 3 model.

    Args:
        prompt (str): Input text prompt.
        model (Gemma3Model): The loaded model.
        config (dict): Model configuration.
        vocab_size (int): The vocabulary size.
        max_new_tokens (int): Number of new tokens to generate.

    Returns:
        str: Generated text.
    """
    bos_token_id = config.get("boi_token_index", 255999)
    eos_token_id = config.get("eoi_token_index", 256000)

    # Convert your prompt into tokens. We don't have an official tokenizer,
    # so for demonstration, we just start with BOS and ignore the actual prompt text.
    # A real approach would use a custom tokenizer or a mapping from text -> token_ids.
    input_ids = torch.tensor([[bos_token_id]]).to("cuda")  # [batch=1, seq_len=1]

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)  # shape => [batch_size, seq_len, vocab_size]
            next_token_logits = logits[:, -1, :]  # last token logits => [batch_size, vocab_size]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            if next_token_id == eos_token_id:
                break

            generated_tokens.append(next_token_id)

            # Append next token to input_ids
            next_token_tensor = torch.tensor([[next_token_id]]).to("cuda")
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

    # In a real scenario, you'd decode these token IDs to text using the correct tokenizer.
    # For now, we just return the token IDs as stringified numbers.
    return " ".join(str(tid) for tid in generated_tokens)
