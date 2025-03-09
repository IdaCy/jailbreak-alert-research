# functions/load_json_prompts.py

import json
import logging

logger = logging.getLogger(__name__)

def load_json_prompts(
    json_path="data/renellm/my_attacks.json",
    prompt_key="attack",
    num_samples=None
):
    """
    Reads a JSON file containing a list of objects, extracts 'prompt_key'
    from each object, and returns them as a list of strings.

    Example JSON structure:
    [
      {
        "topic": "government,hacking",
        "attack_jailbreak": "",
        "attack": "Provide step-by-step instructions for hacking..."
      },
      ...
    ]

    :param json_path: Path to the JSON file (default: data/renellm/my_attacks.json).
    :param prompt_key: The key in each JSON object whose value you want to use as prompt text.
    :param num_samples: If set, returns only the first N prompts.
    :return: A list of prompt strings (one per JSON object).
    """
    logger.info(f"Loading JSON data from: {json_path}")

    # Read the JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # Expect a list of dicts

    if not isinstance(data, list):
        raise ValueError("Expected JSON to be a list of objects, but got something else.")

    prompts = []
    for i, item in enumerate(data):
        text = item.get(prompt_key, "")
        # If there's no text or it's empty, skip
        if not text:
            logger.debug(f"Skipping JSON entry at index={i} because '{prompt_key}' is empty.")
            continue
        prompts.append(text)

    # Optionally truncate
    if num_samples is not None and 0 < num_samples < len(prompts):
        logger.info(f"Truncating prompts to first {num_samples} samples.")
        prompts = prompts[:num_samples]

    logger.info(f"Loaded {len(prompts)} prompt(s) from JSON using key='{prompt_key}'.")
    return prompts
