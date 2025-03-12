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
    A simple transformer-based model for Gemma 3, built based on the config.json parameters.
    """

    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config["hidden_size"],
                nhead=8
            ) for _ in range(config["num_hidden_layers"])
        ])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"])

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

    # Load weights into model
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
    input_ids = torch.tensor([[config["bos_token_id"]]]).to("cuda")  # Start with BOS token
    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).item()
            if next_token == config["eos_token_id"]:
                break
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to("cuda")], dim=1)

    return " ".join([str(token) for token in generated])
