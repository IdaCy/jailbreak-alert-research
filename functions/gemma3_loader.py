import os
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

def download_gemma3_model(model_repo="google/gemma-3-4b-it"):
    """
    Downloads the Gemma 3 model from Hugging Face and loads its weights and configuration.
    
    Returns:
        model_path (str): The path where the model files are stored.
        weights (dict): The model weights loaded from safetensors.
        config (dict): The model configuration.
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

    # Load config.json
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    print("Loaded Model Config:", config)
    
    return model_path, weights, config


class Gemma3Model(nn.Module):
    """
    A simplified transformer-based model for Gemma 3.
    """

    def __init__(self, config):
        super().__init__()

        # Extract vocab_size from token indices (since vocab_size isn't in config)
        vocab_size = config.get("eoi_token_index", 256000) + 1  # Assuming this represents vocab range

        # Extract hidden_size & num_layers
        text_config = config.get("text_config", {})
        hidden_size = text_config.get("hidden_size", 2560)
        num_hidden_layers = text_config.get("num_hidden_layers", 34)

        # Set up model components
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=16,  # Usually, nhead is sqrt(hidden_size) // 2
                dim_feedforward=text_config.get("intermediate_size", 10240),
                activation="gelu"
            ) for _ in range(num_hidden_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def load_gemma3_model(model_repo="google/gemma-3-4b-it"):
    """
    Downloads and initializes the Gemma 3 model.
    
    Returns:
        model (Gemma3Model): The initialized model with loaded weights.
        config (dict): The model configuration.
    """
    model_path, weights, config = download_gemma3_model(model_repo)

    # Create model
    model = Gemma3Model(config)

    # Load weights into model (strict=False to ignore mismatches)
    model.load_state_dict(weights, strict=False)
    model.eval().to("cuda")

    return model, config


def generate_text(prompt, model, config, max_new_tokens=50):
    """
    Generates text using the Gemma 3 model.

    Args:
        prompt (str): Input text prompt.
        model (Gemma3Model): The loaded model.
        config (dict): Model configuration.
        max_new_tokens (int): Number of new tokens to generate.

    Returns:
        str: Generated text.
    """
    bos_token_id = config.get("boi_token_index", 255999)
    eos_token_id = config.get("eoi_token_index", 256000)

    input_ids = torch.tensor([[bos_token_id]]).to("cuda")  # Start with BOS token
    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).item()
            if next_token == eos_token_id:
                break
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to("cuda")], dim=1)

    return " ".join([str(token) for token in generated])
