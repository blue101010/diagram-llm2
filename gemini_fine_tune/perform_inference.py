import os
import json
import time
import random
import logging
import google.generativeai as genai
from google import genai as vertex_genai
from google.genai import types
from typing import Dict, List, Any
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv("GOOGLE_API_KEY")  # For the base model
BASE_MODEL = "gemini-2.0-flash-001"
FINE_TUNED_MODEL_ID = os.getenv("FINE_TUNED_MODEL_ID")
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")

OUTPUT_DIR = "outputs"
RAW_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_responses.json")
CLEAN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clean_responses.json")
VALIDATION_DATA_FILE = "validation_data.jsonl"

# Retry Configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0


def setup() -> None:
    """Initialize the environment."""
    genai.configure(api_key=API_KEY)

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def load_validation_data(file_path: str) -> List[Dict[str, Any]]:
    """Load validation data from JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data


def generate_base_model_response(prompt: str, system_instruction: str) -> str:
    """Generate a response from the base model."""
    return ""


def generate_fine_tuned_response(prompt: str) -> str:
    """Generate a response from the fine-tuned model using Vertex AI with retry logic."""
    client = vertex_genai.Client(
        vertexai=True,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_LOCATION,
    )

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="zephyr")
            ),
        ),
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
        ],
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = ""
            for chunk in client.models.generate_content_stream(
                model=FINE_TUNED_MODEL_ID,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    response += chunk.text
            return response

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Error generating response from fine-tuned model after {MAX_RETRIES} attempts: {e}")
                return f"Error: {str(e)}"
            
            delay = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"API Error: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)
    
    return "Error: Max retries exceeded"


def extract_mermaid_code(response: str) -> str:
    """Extract mermaid code from the response."""
    if "```mermaid" in response:
        start_idx = response.find("```mermaid") + len("```mermaid")
        end_idx = response.find("```", start_idx)
        if end_idx != -1:
            return response[start_idx:end_idx].strip()

    # If no mermaid code block is found, return the entire response
    return response.strip()


def main() -> None:
    """Main execution function."""
    setup()

    # Load validation data
    logger.info(f"Loading validation data from {VALIDATION_DATA_FILE}...")
    validation_data = load_validation_data(VALIDATION_DATA_FILE)
    logger.info(f"Loaded {len(validation_data)} validation examples.")

    # Load existing results if available to resume progress
    all_results = []
    if os.path.exists(RAW_OUTPUT_FILE):
        try:
            with open(RAW_OUTPUT_FILE, "r") as f:
                all_results = json.load(f)
            logger.info(f"Resuming from {len(all_results)} existing results.")
        except json.JSONDecodeError:
            logger.warning("Could not load existing results, starting fresh.")

    start_index = len(all_results)

    for i, example in enumerate(validation_data[start_index:], start=start_index):
        logger.info(f"Processing example {i+1}/{len(validation_data)}...")

        # Extract prompt and system instruction
        system_instruction = example["systemInstruction"]["parts"][0]["text"]
        prompt = example["contents"][0]["parts"][0]["text"]
        expected_output = example["contents"][1]["parts"][0]["text"]

        result = {"prompt": prompt, "expected_output": expected_output, "models": {}}

        # Generate response from base model
        logger.info(f"Generating response from {BASE_MODEL}...")
        base_response = generate_base_model_response(prompt, system_instruction)
        base_clean = extract_mermaid_code(base_response)

        result["models"][BASE_MODEL] = {
            "raw_response": base_response,
            "clean_mermaid": base_clean,
        }

        # Generate response from fine-tuned model
        logger.info(f"Generating response from fine-tuned model...")
        ft_response = generate_fine_tuned_response(prompt)
        ft_clean = extract_mermaid_code(ft_response)
        
        result["models"]["fine-tuned"] = {
            "raw_response": ft_response,
            "clean_mermaid": ft_clean,
        }

        all_results.append(result)

        # Save progress incrementally (every 5 items or at the end)
        if (i + 1) % 5 == 0 or (i + 1) == len(validation_data):
            logger.info(f"Saving progress to {RAW_OUTPUT_FILE}...")
            with open(RAW_OUTPUT_FILE, "w") as f:
                json.dump(all_results, f, indent=2)

    # Save clean responses (just the extracted mermaid code)

    # Save clean responses (just the extracted mermaid code)
    clean_results = []
    for result in all_results:
        clean_result = {
            "prompt": result["prompt"],
            "expected_output": result["expected_output"],
            "models": {
                BASE_MODEL: result["models"][BASE_MODEL]["clean_mermaid"],
                "fine-tuned": result["models"]["fine-tuned"]["clean_mermaid"],
            },
        }
        clean_results.append(clean_result)

    with open(CLEAN_OUTPUT_FILE, "w") as f:
        json.dump(clean_results, f, indent=2)

    logger.info(f"All results saved to {RAW_OUTPUT_FILE} and {CLEAN_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
