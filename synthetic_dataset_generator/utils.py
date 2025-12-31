import json
import re
import time
import random
import os
from typing import Any, Dict, Optional
import google.generativeai as genai

from synthetic_dataset_generator.config import (
    logger,
    MAX_RETRIES,
    INITIAL_BACKOFF,
    OUTPUT_FILE
)

def safe_json_loads(text: str) -> Any:
    """Parse JSON robustly by extracting the JSON substring if necessary."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the JSON object using a regular expression
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                logger.error(f"[safe_json_loads] Regex extraction failed: {e}", exc_info=True)
                raise
        else:
            raise


def call_gemini_with_retry(
    model: Any,
    prompt: str,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Call Gemini API with exponential backoff retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            return model.generate_content(prompt, generation_config=generation_config)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"[call_gemini_with_retry] Failed after {MAX_RETRIES} attempts: {e}")
                raise e
            
            delay = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"[call_gemini_with_retry] API Error: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)


def append_to_output_file(entry: Dict[str, str]) -> None:
    """Append a single entry to the output file."""
    try:
        # Read existing data
        existing_data = []
        if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[append_to_output_file] Error parsing {OUTPUT_FILE}, starting fresh"
                    )
                    existing_data = []

        # Append new entry
        existing_data.append(entry)

        # Write back to file
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)

    except Exception as e:
        logger.error(f"[append_to_output_file] Error updating output file: {e}", exc_info=True)
