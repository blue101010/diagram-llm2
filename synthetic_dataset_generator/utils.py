import json
import re
import time
import random
import os
from typing import Any, Dict, Optional
from google import genai

from synthetic_dataset_generator.config import (
    logger,
    MAX_RETRIES,
    INITIAL_BACKOFF,
    OUTPUT_FILE,
    RATE_LIMIT_DELAY,
    MODELS_LIMITS_FILE
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
    client: genai.Client,
    model_id: str,
    prompt: str,
    config: Optional[Any] = None,
) -> Optional[Any]:
    """Call Gemini API with exponential backoff retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            if RATE_LIMIT_DELAY > 0:
                time.sleep(RATE_LIMIT_DELAY)

            return client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=config
            )
        except Exception as e:
            error_str = str(e)
            
            if attempt == MAX_RETRIES - 1:
                logger.error(f"[call_gemini_with_retry] Failed after {MAX_RETRIES} attempts: {e}")
                raise e

            # Check for 429 or Resource Exhausted
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Try to extract retry delay from message
                # Pattern: "Please retry in 58.310634397s."
                match = re.search(r"retry in (\d+(\.\d+)?)s", error_str)
                if match:
                    delay = float(match.group(1)) + 1.0 # Add buffer
                    logger.warning(f"[call_gemini_with_retry] Rate limit hit [Attempt {attempt+1}/{MAX_RETRIES}]. Waiting {delay:.2f}s as requested by API...")
                else:
                    delay = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"[call_gemini_with_retry] Rate limit hit [Attempt {attempt+1}/{MAX_RETRIES}]. Retrying in {delay:.2f}s...")
            else:
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

def load_model_limits() -> Dict[str, Any]:
    """Load model limits from the JSON file."""
    try:
        if os.path.exists(MODELS_LIMITS_FILE):
            with open(MODELS_LIMITS_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.warning(f"[load_model_limits] Failed to load {MODELS_LIMITS_FILE}: {e}")
        return {}


def select_model(default_model: str = "gemma-3-27b-it") -> str:
    """Interactive model selection."""
    models_limits = load_model_limits()
    
    # Get list of models, ensuring default is present even if not in file (though it should be)
    models = list(models_limits.keys())
    if not models:
        # Fallback list if file missing or empty
        models = [
            "gemma-3-27b-it",
            "gemini-2.5-flash", 
            "gemini-2.0-flash", 
            "gemini-2.0-flash-lite"
        ]
    
    print("\nSelect Model:")
    print(f"{'ID':<4} {'Model Name':<40} {'RPM Limit':<10} {'Note'}")
    print("-" * 80)
    
    for i, model in enumerate(models):
        note = ""
        rpm_info = "N/A"
        
        if model in models_limits:
            rpm = models_limits[model].get("rpm", 0)
            rpm_info = str(rpm)
            if rpm < 10:
                note = "(Low RPM)"
        
        if model == default_model:
            note = f"{note} (Default/Recommended)".strip()

        print(f"{i + 1:<4} {model:<40} {rpm_info:<10} {note}")
    
    choice = input(f"\nEnter choice [1-{len(models)}] (Press Enter for {default_model}): ").strip()
    
    selected_model = default_model
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected_model = models[idx]
            else:
                print("Invalid choice. Using default.")
        except ValueError:
            print("Invalid input. Using default.")

    # Update global rate limit if possible (hacky but effective for this script structure)
    # Ideally should return it, but config is imported everywhere.
    # We will update the config module's variable directly in main.
    
    return selected_model
