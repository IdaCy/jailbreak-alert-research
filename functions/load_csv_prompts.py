# functions/load_csv_prompts.py
import logging
import pandas as pd

def load_prompts(file_path, num_samples=None, logger=None):
    """
    Reads a file line-by-line into a list of strings. 
    Optionally truncates to `num_samples`.
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    logger.info(f"Loading lines from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if num_samples is not None and 0 < num_samples < len(lines):
        logger.info(f"Truncating dataset to first {num_samples} lines.")
        lines = lines[:num_samples]

    logger.info(f"Loaded {len(lines)} prompts from {file_path}.")
    return lines
