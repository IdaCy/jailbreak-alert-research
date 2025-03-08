# functions/load_csv_prompts.py
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_prompts(csv_path, num_samples=None):
    """
    Load prompt lines from a file and return them as a list of strings.
    
    :param csv_path: Path to the file containing prompts (CSV or plain text).
    :param num_samples: Optionally truncate to this number of samples.
    :return: A list of prompt strings.
    """
    logger.info(f"Loading prompts from: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if num_samples is not None and num_samples < len(lines):
        lines = lines[:num_samples]
        logger.info(f"Truncated prompts to {num_samples} lines.")

    logger.info(f"Loaded {len(lines)} lines from {csv_path}")
    return lines
