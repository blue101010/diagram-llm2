import json
import re
import random
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Function to remove leading numbers (like "1. " or "71. ") from prompts
def clean_prompt(prompt: str) -> str:
    return re.sub(r"^\d+\.\s+", "", prompt)


# Function to convert a dictionary to the Gemini fine-tuning format
def to_fine_tuning_format(entry: Dict[str, Any]) -> Dict[str, Any]:
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

try:
    # Load the main dataset
    with open("FINAL_DATASET_MAIN(1).json", "r") as f:
        main_data = json.load(f)

    # Load the side dataset for validation
    with open("side_dataset.json", "r") as f:
        side_data = json.load(f)

    # Select 150 random entries from the side dataset for validation
    random.seed(42)  # For reproducibility
    validation_data = random.sample(side_data, 150)

    # Convert all data to the fine-tuning format
    main_data_formatted = [to_fine_tuning_format(entry) for entry in main_data]
    validation_data_formatted = [to_fine_tuning_format(entry) for entry in validation_data]

    # Shuffle both datasets to avoid any sequence-related biases
    random.shuffle(main_data_formatted)
    random.shuffle(validation_data_formatted)

    # Write the training data to a JSONL file
    with open("training_data.jsonl", "w") as f:
        for entry in main_data_formatted:
            f.write(json.dumps(entry) + "\n")

    # Write the validation data to a JSONL file
    with open("validation_data.jsonl", "w") as f:
        for entry in validation_data_formatted:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Conversion complete!")
    logger.info(
        f"Training data: {len(main_data_formatted)} examples written to training_data.jsonl"
    )
    logger.info(
        f"Validation data: {len(validation_data_formatted)} examples written to validation_data.jsonl"
    )

except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
except json.JSONDecodeError as e:
    logger.error(f"Error decoding JSON: {e}")
except Exception as e:
    logger.error(f"An unexpected error occurred: {e}", exc_info=True)
