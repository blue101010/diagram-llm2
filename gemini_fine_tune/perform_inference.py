import os
import json
import time
import random
import logging
import re
from google import genai
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
# BASE_MODEL will be selected by user
FINE_TUNED_MODEL_ID = os.getenv("FINE_TUNED_MODEL_ID")
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")

AVAILABLE_MODELS = [
    "gemma-3-27b-it",
    "gemini-3-flash-preview",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    "gemma-3-1b-it",
    "gemini-2.5-flash-preview-tts",
    "gemini-robotics-er-1.5-preview",
    "gemini-2.5-flash-native-audio-latest"
]

# Rate Limiting Configuration
# Will be updated dynamically based on model selection and models_limits_free.json
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "15.0"))

MODELS_LIMITS_FILE = "models_limits_free.json"
MODELS_LIMITS = {}

# Load model limits if file exists
if os.path.exists(MODELS_LIMITS_FILE):
    try:
        with open(MODELS_LIMITS_FILE, "r") as f:
            MODELS_LIMITS = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {MODELS_LIMITS_FILE}: {e}")
elif os.path.exists(os.path.join("gemini_fine_tune", MODELS_LIMITS_FILE)):
    try:
        with open(os.path.join("gemini_fine_tune", MODELS_LIMITS_FILE), "r") as f:
            MODELS_LIMITS = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {MODELS_LIMITS_FILE}: {e}")

OUTPUT_DIR = "outputs"
RAW_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_responses.json")
CLEAN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clean_responses.json")

# Handle path relative to script location or current working directory
if os.path.exists(os.path.join("gemini_fine_tune", "dataset", "validation_data.jsonl")):
    VALIDATION_DATA_FILE = os.path.join("gemini_fine_tune", "dataset", "validation_data.jsonl")
else:
    VALIDATION_DATA_FILE = os.path.join("dataset", "validation_data.jsonl")

# Retry Configuration
MAX_RETRIES = 10  # Increased retries for 429s
INITIAL_BACKOFF = 2.0


def setup() -> None:
    """Initialize the environment."""
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


def select_base_model() -> str:
    """Prompt user to select a base model and update RATE_LIMIT_DELAY."""
    global RATE_LIMIT_DELAY
    
    print("\nSelect Base Model:")
    print(f"{'ID':<4} {'Model Name':<40} {'RPM Limit':<10} {'Note'}")
    print("-" * 80)
    
    for i, model in enumerate(AVAILABLE_MODELS):
        note = ""
        rpm_info = "N/A"
        
        if model in MODELS_LIMITS:
            rpm = MODELS_LIMITS[model].get("rpm", 0)
            rpm_info = str(rpm)
            if rpm < 10:
                note = "(Low RPM)"
        
        if i == 0:
            note = "(Recommended / Default)"
        elif "gemini-3-flash" in model:
            note = "(Low Quota: 20 RPD)"
        
        print(f"{i + 1:<4} {model:<40} {rpm_info:<10} {note}")
    
    choice = input(f"\nEnter choice [1-{len(AVAILABLE_MODELS)}] (Press Enter for {AVAILABLE_MODELS[0]}): ").strip()
    
    selected_model = AVAILABLE_MODELS[0]
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(AVAILABLE_MODELS):
                selected_model = AVAILABLE_MODELS[idx]
            else:
                print("Invalid choice. Using default.")
        except ValueError:
            print("Invalid input. Using default.")
    
    # Update RATE_LIMIT_DELAY based on selected model
    if selected_model in MODELS_LIMITS:
        rpm = MODELS_LIMITS[selected_model].get("rpm", 0)
        if rpm > 0:
            # Calculate delay: 60 / (RPM - 1) to be safe
            # If RPM is very high (e.g. 9999), delay is negligible
            target_rpm = max(1, rpm - 1)
            new_delay = 60.0 / target_rpm
            print(f"\n[Config] Selected model: {selected_model}")
            print(f"[Config] Max RPM: {rpm}. Setting target RPM to {target_rpm}.")
            print(f"[Config] Updating RATE_LIMIT_DELAY from {RATE_LIMIT_DELAY}s to {new_delay:.2f}s")
            RATE_LIMIT_DELAY = new_delay
    
    return selected_model


def generate_base_model_response(model_name: str, prompt: str, system_instruction: str) -> str:
    """Generate a response from the base model."""
    
    # Gemma models might not support system_instruction in config (Error 400)
    # Workaround: Prepend to prompt
    if "gemma" in model_name.lower():
        if system_instruction:
            prompt = f"System Instruction: {system_instruction}\n\nUser Prompt: {prompt}"
            system_instruction = None

    # Try v1beta first (default), then v1alpha if 404
    api_versions = ["v1beta", "v1alpha"]
    
    for api_version in api_versions:
        client = genai.Client(api_key=API_KEY, http_options={'api_version': api_version})
        
        config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            system_instruction=system_instruction
        )

        for attempt in range(MAX_RETRIES):
            try:
                if RATE_LIMIT_DELAY > 0:
                    time.sleep(RATE_LIMIT_DELAY)

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                
                # If 404 Not Found, break retry loop and try next API version
                if "404" in error_str and "NOT_FOUND" in error_str:
                    logger.warning(f"Model {model_name} not found in {api_version}. Trying next version if available...")
                    break # Break inner loop to try next api_version
                
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Error generating response from base model after {MAX_RETRIES} attempts: {e}")
                    return f"Error: {str(e)}"
                
                # Check for 429 or Resource Exhausted
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Try to extract retry delay from message
                    # Pattern: "Please retry in 58.310634397s."
                    match = re.search(r"retry in (\d+(\.\d+)?)s", error_str)
                    if match:
                        delay = float(match.group(1)) + 1.0 # Add buffer
                        logger.warning(f"Rate limit hit [Attempt {attempt+1}/{MAX_RETRIES}]. Waiting {delay:.2f}s as requested by API...")
                    else:
                        delay = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Rate limit hit [Attempt {attempt+1}/{MAX_RETRIES}]. Retrying in {delay:.2f}s...")
                else:
                    delay = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Base Model API Error: {e} [Attempt {attempt+1}/{MAX_RETRIES}]. Retrying in {delay:.2f}s...")
                
                time.sleep(delay)
        
        # If we successfully returned, we wouldn't be here.
        # If we broke out of the loop due to 404, we continue to next api_version.
        # If we exhausted retries, we returned Error.
    
    return f"Error: Model {model_name} not found or failed in all API versions."


def generate_fine_tuned_response(prompt: str) -> str:
    """Generate a response from the fine-tuned model using Vertex AI with retry logic."""
    # If FINE_TUNED_MODEL_ID starts with "projects/", it's a Vertex AI endpoint.
    # Otherwise, treat it as a standard Gemini model ID (e.g., "tunedModels/...")
    
    is_vertex = FINE_TUNED_MODEL_ID.startswith("projects/")
    
    if is_vertex:
        try:
            client = genai.Client(
                vertexai=True,
                project=VERTEX_PROJECT_ID,
                location=VERTEX_LOCATION,
            )
        except ValueError as e:
            logger.error(f"Failed to initialize Vertex AI client: {e}")
            return f"Error: {str(e)}"
    else:
        # Use standard API Key client for non-Vertex tuned models
        client = genai.Client(api_key=API_KEY)

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        # Speech config removed as it might not be supported on all models/endpoints
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
            if RATE_LIMIT_DELAY > 0:
                time.sleep(RATE_LIMIT_DELAY)

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
            error_str = str(e)
            
            # Check for Missing Credentials (ADC)
            if "default credentials were not found" in error_str:
                logger.error("Vertex AI Authentication Failed: Application Default Credentials (ADC) not found.")
                logger.error("To fix this, run: 'gcloud auth application-default login' in your terminal.")
                logger.error("Or set GOOGLE_APPLICATION_CREDENTIALS to your service account key path.")
                return "Error: Vertex AI Authentication Failed (ADC not found)."

            if attempt == MAX_RETRIES - 1:
                logger.error(f"Error generating response from fine-tuned model after {MAX_RETRIES} attempts: {e}")
                return f"Error: {str(e)}"
            
            # Check for 429 or Resource Exhausted
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                match = re.search(r"retry in (\d+(\.\d+)?)s", error_str)
                if match:
                    delay = float(match.group(1)) + 1.0
                    logger.warning(f"Rate limit hit (Fine-tuned) [Attempt {attempt+1}/{MAX_RETRIES}]. Waiting {delay:.2f}s as requested by API...")
                else:
                    delay = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit (Fine-tuned) [Attempt {attempt+1}/{MAX_RETRIES}]. Retrying in {delay:.2f}s...")
            else:
                delay = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"API Error: {e} [Attempt {attempt+1}/{MAX_RETRIES}]. Retrying in {delay:.2f}s...")
            
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
    
    # Select model first to display in banner
    base_model = select_base_model()

    print("=" * 60)
    print("       GEMINI FINE-TUNE INFERENCE & VALIDATION SCRIPT")
    print("=" * 60)
    print(f"Description:  Generates responses from Base Model ({base_model})")
    print(f"              and Fine-Tuned Model ({FINE_TUNED_MODEL_ID})")
    print(f"              for comparison using validation dataset.")
    print("-" * 60)
    print(f"Input Data:   {VALIDATION_DATA_FILE}")
    print(f"Output Raw:   {RAW_OUTPUT_FILE}")
    print(f"Output Clean: {CLEAN_OUTPUT_FILE}")
    print("-" * 60)
    print(f"Rate Limit:   {RATE_LIMIT_DELAY}s delay between requests")
    if RATE_LIMIT_DELAY >= 10.0:
        print("              (Optimized for STRICT FREE-TIER: ~4 RPM)")
        print("              Set RATE_LIMIT_DELAY=0.0 for higher tiers.")
    elif RATE_LIMIT_DELAY >= 4.0:
        print("              (Optimized for FREE-TIER: ~15 RPM)")
        print("              Set RATE_LIMIT_DELAY=0.0 for higher tiers.")
    print(f"Max Retries:  {MAX_RETRIES}")
    print("=" * 60)
    print("\n")

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
        logger.info(f"Generating response from {base_model}...")
        base_response = generate_base_model_response(base_model, prompt, system_instruction)
        base_clean = extract_mermaid_code(base_response)

        result["models"][base_model] = {
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
                base_model: result["models"][base_model]["clean_mermaid"],
                "fine-tuned": result["models"]["fine-tuned"]["clean_mermaid"],
            },
        }
        clean_results.append(clean_result)

    with open(CLEAN_OUTPUT_FILE, "w") as f:
        json.dump(clean_results, f, indent=2)

    logger.info(f"All results saved to {RAW_OUTPUT_FILE} and {CLEAN_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
