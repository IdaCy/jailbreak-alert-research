import os
import logging
import pandas as pd
import logging

def load_csv_prompts(file_path, logger=None, num_samples=None):
    """
    Reads prompts from a file line by line. Ignores commas, 
    since each prompt is newline-separated. Returns a list of 
    prompt strings. If num_samples is provided, truncates to that many.
    """
    if logger is None:
        logger = logging.getLogger("load_csv_prompts")
        logger.setLevel(logging.INFO)

    logger.debug(f"Loading prompts from {file_path}")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read line-by-line, ignoring commas
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Truncate if num_samples is specified
    if num_samples is not None and isinstance(num_samples, int) and num_samples > 0:
        lines = lines[:num_samples]
        logger.debug(f"Truncated to first {num_samples} prompts.")

    logger.debug(f"Total lines loaded: {len(lines)}")
    # If need DataFrame, can wrap:
    # df = pd.DataFrame(lines, columns=["sentence"])
    # return df["sentence"].tolist()
    return lines
