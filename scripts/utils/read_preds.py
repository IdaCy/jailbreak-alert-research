import os
import glob
import torch
import logging
from datetime import datetime
from tqdm import tqdm

# ------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------
READ_OUTPUT_DIR = (
    os.environ.get("READ_OUTPUT_DIR")
    or globals().get("READ_OUTPUT_DIR")
    or "output/extractions/mistral7b/jb"
)

READ_LOG_FILE = (
    os.environ.get("READ_LOG_FILE")
    or globals().get("READ_LOG_FILE")
    or "logs/read_predictions.log"
)

MAX_PREDICTIONS = (
    os.environ.get("MAX_PREDICTIONS")
    or globals().get("MAX_PREDICTIONS")
    or 20
)
# Ensure MAX_PREDICTIONS is an integer
if isinstance(MAX_PREDICTIONS, str) and MAX_PREDICTIONS.isdigit():
    MAX_PREDICTIONS = int(MAX_PREDICTIONS)

WRITE_PREDICTIONS_FILE = (
    os.environ.get("WRITE_PREDICTIONS_FILE")
    or globals().get("WRITE_PREDICTIONS_FILE")
    or "logs/pred_out.txt"
)

# Create logs directory if needed
os.makedirs(os.path.dirname(READ_LOG_FILE), exist_ok=True)

# ------------------------------------------------------------------------
# 1a. Set up logging
# ------------------------------------------------------------------------
logger = logging.getLogger("ReadPredictionsLogger")
logger.setLevel(logging.DEBUG)  # capture everything

# File handler
file_handler = logging.FileHandler(READ_LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

logger.info("=== Starting read_predictions script ===")
logger.info(f"READ_OUTPUT_DIR = {READ_OUTPUT_DIR}")
logger.info(f"READ_LOG_FILE = {READ_LOG_FILE}")
logger.info(f"MAX_PREDICTIONS = {MAX_PREDICTIONS}")
logger.info(f"WRITE_PREDICTIONS_FILE = {WRITE_PREDICTIONS_FILE}")

# ------------------------------------------------------------------------
# 2. Locate and sort the .pt files
# ------------------------------------------------------------------------
pt_files = sorted(glob.glob(os.path.join(READ_OUTPUT_DIR, "activations_*.pt")))
if not pt_files:
    logger.warning(f"No .pt files found in {READ_OUTPUT_DIR}. Exiting.")
    exit(0)

logger.info(f"Found {len(pt_files)} .pt files to process.")

# ------------------------------------------------------------------------
# 3. Read predictions from each file, optionally write to a file
# ------------------------------------------------------------------------
all_predictions = []
predictions_collected = 0

for pt_file in tqdm(pt_files, desc="Reading .pt files"):
    logger.debug(f"Loading file: {pt_file}")
    try:
        data = torch.load(pt_file, map_location="cpu")
        # The dictionary from your script includes:
        #  "final_predictions": List of strings
        if "final_predictions" in data:
            for pred in data["final_predictions"]:
                all_predictions.append(pred)
                predictions_collected += 1
                if predictions_collected >= MAX_PREDICTIONS:
                    logger.info("Reached MAX_PREDICTIONS limit; stopping.")
                    break
        else:
            logger.warning(f"No 'final_predictions' key in {pt_file}.")

    except Exception as e:
        logger.exception(f"Could not load {pt_file}: {str(e)}")

    if predictions_collected >= MAX_PREDICTIONS:
        break

logger.info(f"Collected {len(all_predictions)} total predictions.")

# ------------------------------------------------------------------------
# 4. Print predictions and optionally write them to a file
# ------------------------------------------------------------------------
logger.info("=== Sample of collected predictions ===")
for i, prediction in enumerate(all_predictions):
    if i < 5:  # Show only first few on console
        logger.info(f"Prediction {i+1}: {prediction}")
    else:
        break

# If user wants to write predictions to a file:
if WRITE_PREDICTIONS_FILE:
    os.makedirs(os.path.dirname(WRITE_PREDICTIONS_FILE), exist_ok=True)
    logger.info(f"Writing all predictions to {WRITE_PREDICTIONS_FILE}")
    try:
        with open(WRITE_PREDICTIONS_FILE, "w", encoding="utf-8") as out_f:
            for pred in all_predictions:
                out_f.write(pred.strip() + "\n")
        logger.info("Finished writing predictions.")
    except Exception as e:
        logger.exception(f"Error writing predictions file: {str(e)}")

logger.info("=== read_predictions script complete ===")
