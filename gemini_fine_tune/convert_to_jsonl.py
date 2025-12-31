import json
import re
import random
import logging
import os
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MAIN_DATASET_FILE = "FINAL_DATASET_MAIN(1).json"
SIDE_DATASET_FILE = "side_dataset.json"
TRAINING_OUTPUT_FILE = "training_data.jsonl"
VALIDATION_OUTPUT_FILE = "validation_data.jsonl"


# Function to remove leading numbers (like "1. " or "71. ") from prompts
def clean_prompt(prompt: str) -> str:
    return re.sub(r"^\d+\.\s+", "", prompt)


def validate_entry(entry: Dict[str, Any]) -> bool:
    """Validate a single dataset entry."""
    if not isinstance(entry, dict):
        logger.warning(f"Skipping invalid entry type: {type(entry)}")
        return False
    
    if "prompt" not in entry or "output" not in entry:
        logger.warning(f"Skipping entry missing keys. Keys found: {list(entry.keys())}")
        return False
        
    if not isinstance(entry["prompt"], str) or not isinstance(entry["output"], str):
        logger.warning("Skipping entry with non-string values")
        return False
        
    if not entry["prompt"].strip():
        logger.warning("Skipping entry with empty prompt")
        return False
        
    if not entry["output"].strip():
        logger.warning("Skipping entry with empty output")
        return False
        
    return True


# Function to convert a dictionary to the Gemini fine-tuning format
def to_fine_tuning_format(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not validate_entry(entry):
        return None

    clean_prompt_text = clean_prompt(entry["prompt"])
    output_text = entry["output"]

    return {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You are a Mermaid diagram generator that creates high-quality diagrams from textual descriptions. You should only give the mermaid answer for the particular question."
                }
            ],
        },
        "contents": [
            {"role": "user", "parts": [{"text": clean_prompt_text}]},
            {"role": "model", "parts": [{"text": output_text}]},
        ],
    }

def load_and_validate_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from file and validate existence."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    with open(file_path, "r") as f:
        data = json.load(f)
        
    if not isinstance(data, list):
        raise ValueError(f"Dataset in {file_path} must be a list of objects")
        
    return data

try:
    # Load the main dataset
    logger.info(f"Loading main dataset from {MAIN_DATASET_FILE}...")
    main_data = load_and_validate_dataset(MAIN_DATASET_FILE)

    # Load the side dataset for validation
    logger.info(f"Loading side dataset from {SIDE_DATASET_FILE}...")
    side_data = load_and_validate_dataset(SIDE_DATASET_FILE)

    # Select 150 random entries from the side dataset for validation
    random.seed(42)  # For reproducibility
    if len(side_data) < 150:
        logger.warning(f"Side dataset has fewer than 150 entries ({len(side_data)}). Using all available.")
        validation_data = side_data
    else:
        validation_data = random.sample(side_data, 150)

    # Convert all data to the fine-tuning format
    main_data_formatted = []
    for entry in main_data:
        formatted = to_fine_tuning_format(entry)
        if formatted:
            main_data_formatted.append(formatted)

    validation_data_formatted = []
    for entry in validation_data:
        formatted = to_fine_tuning_format(entry)
        if formatted:
            validation_data_formatted.append(formatted)

    # Shuffle both datasets to avoid any sequence-related biases
    random.shuffle(main_data_formatted)
    random.shuffle(validation_data_formatted)

    # Write the training data to a JSONL file
    logger.info(f"Writing training data to {TRAINING_OUTPUT_FILE}...")
    with open(TRAINING_OUTPUT_FILE, "w") as f:
        for entry in main_data_formatted:
            f.write(json.dumps(entry) + "\n")

    # Write the validation data to a JSONL file
    logger.info(f"Writing validation data to {VALIDATION_OUTPUT_FILE}...")
    with open(VALIDATION_OUTPUT_FILE, "w") as f:
        for entry in validation_data_formatted:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Conversion complete!")
    logger.info(
        f"Training data: {len(main_data_formatted)} examples written to {TRAINING_OUTPUT_FILE}"
    )
    logger.info(
        f"Validation data: {len(validation_data_formatted)} examples written to {VALIDATION_OUTPUT_FILE}"
    )

except FileNotFoundError as e:
    logger.error(f"File error: {e}")
except json.JSONDecodeError as e:
    logger.error(f"Error decoding JSON: {e}")
except ValueError as e:
    logger.error(f"Data validation error: {e}")
except Exception as e:
    logger.error(f"An unexpected error occurred: {e}", exc_info=True)
