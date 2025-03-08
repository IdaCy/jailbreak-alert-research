# myproject/read_predictions.py

import os
import glob
import torch
import logging
from datetime import datetime
from tqdm import tqdm

def read_predictions(
    read_output_dir="output/extractions/gemma2b/jb",
    max_predictions=20,
    write_predictions_file=None,
    log_file="logs/read_predictions.log",
):
    """
    Scans a directory of `.pt` files (named like "activations_XXXX_YYYY.pt"),
    loads them up to a max number of predictions, and optionally writes them
    to a text file.

    Returns a list of all collected predictions (strings).
    """
    # Create logs directory if needed
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # ----------------------------------------------------------------
    # Set up logging
    # ----------------------------------------------------------------
    logger = logging.getLogger("ReadPredictionsLogger")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("=== Starting read_predictions function ===")
    logger.info(f"read_output_dir = {read_output_dir}")
    logger.info(f"log_file = {log_file}")
    logger.info(f"max_predictions = {max_predictions}")
    logger.info(f"write_predictions_file = {write_predictions_file}")

    # ----------------------------------------------------------------
    # 2. Locate and sort the .pt files
    # ----------------------------------------------------------------
    pt_files = sorted(glob.glob(os.path.join(read_output_dir, "activations_*.pt")))
    if not pt_files:
        logger.warning(f"No .pt files found in {read_output_dir}. Returning empty list.")
        return []  # Return empty list instead of exit(0)

    logger.info(f"Found {len(pt_files)} .pt files to process.")

    # ----------------------------------------------------------------
    # 3. Read predictions from each file
    # ----------------------------------------------------------------
    all_predictions = []
    predictions_collected = 0

    for pt_file in tqdm(pt_files, desc="Reading .pt files"):
        logger.debug(f"Loading file: {pt_file}")
        try:
            data = torch.load(pt_file, map_location="cpu")
            # The dictionary from your script includes a "final_predictions" key
            if "final_predictions" in data:
                for pred in data["final_predictions"]:
                    all_predictions.append(pred)
                    predictions_collected += 1
                    if predictions_collected >= max_predictions:
                        logger.info("Reached max_predictions limit; stopping.")
                        break
            else:
                logger.warning(f"No 'final_predictions' key in {pt_file}.")
        except Exception as e:
            logger.exception(f"Could not load {pt_file}: {str(e)}")

        if predictions_collected >= max_predictions:
            break

    logger.info(f"Collected {len(all_predictions)} total predictions.")

    # ----------------------------------------------------------------
    # 4. Optionally print sample + write to file
    # ----------------------------------------------------------------
    logger.info("=== Sample of collected predictions ===")
    for i, prediction in enumerate(all_predictions[:5]):  # only first 5
        logger.info(f"Prediction {i+1}: {prediction}")

    if write_predictions_file:
        os.makedirs(os.path.dirname(write_predictions_file), exist_ok=True)
        logger.info(f"Writing all predictions to {write_predictions_file}")
        try:
            with open(write_predictions_file, "w", encoding="utf-8") as out_f:
                for pred in all_predictions:
                    out_f.write(pred.strip() + "\n")
            logger.info("Finished writing predictions.")
        except Exception as e:
            logger.exception(f"Error writing predictions file: {str(e)}")

    logger.info("=== read_predictions function complete ===")

    return all_predictions
