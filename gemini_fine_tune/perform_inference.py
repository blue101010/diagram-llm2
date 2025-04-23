import os
import json
import google.generativeai as genai
from google import genai as vertex_genai
from google.genai import types
from typing import Dict, List, Any

# Constants
API_KEY = ""  # For the base model
BASE_MODEL = "gemini-2.0-flash-001"
FINE_TUNED_MODEL_ID = (
    "projects/1091107469352/locations/us-central1/endpoints/613585762415280128"
)
OUTPUT_DIR = "outputs"
RAW_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_responses.json")
CLEAN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clean_responses.json")
VALIDATION_DATA_FILE = "validation_data.jsonl"


def setup():
    """Initialize the environment."""
    genai.configure(api_key=API_KEY)

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def load_validation_data(file_path: str) -> List[Dict]:
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
    """Generate a response from the fine-tuned model using Vertex AI."""
    try:
        client = vertex_genai.Client(
            vertexai=True,
            project="1091107469352",
            location="us-central1",
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
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                ),
            ],
        )

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
        print(f"Error generating response from fine-tuned model: {e}")
        return f"Error: {str(e)}"


def extract_mermaid_code(response: str) -> str:
    """Extract mermaid code from the response."""
    if "```mermaid" in response:
        start_idx = response.find("```mermaid") + len("```mermaid")
        end_idx = response.find("```", start_idx)
        if end_idx != -1:
            return response[start_idx:end_idx].strip()

    # If no mermaid code block is found, return the entire response
    return response.strip()


def main():
    """Main execution function."""
    setup()

    # Load validation data
    print(f"Loading validation data from {VALIDATION_DATA_FILE}...")
    validation_data = load_validation_data(VALIDATION_DATA_FILE)
    print(f"Loaded {len(validation_data)} validation examples.")

    all_results = []

    for i, example in enumerate(validation_data):
        print(f"\nProcessing example {i+1}/{len(validation_data)}...")

        # Extract prompt and system instruction
        system_instruction = example["systemInstruction"]["parts"][0]["text"]
        prompt = example["contents"][0]["parts"][0]["text"]
        expected_output = example["contents"][1]["parts"][0]["text"]

        result = {"prompt": prompt, "expected_output": expected_output, "models": {}}

        # Generate response from base model
        print(f"Generating response from {BASE_MODEL}...")
        base_response = generate_base_model_response(prompt, system_instruction)
        base_clean = extract_mermaid_code(base_response)

        result["models"][BASE_MODEL] = {
            "raw_response": base_response,
            "clean_mermaid": base_clean,
        }

        # Generate response from fine-tuned model
        # print(f"Generating response from fine-tuned model...")
        # ft_response = generate_fine_tuned_response(prompt)
        # ft_clean = extract_mermaid_code(ft_response)
        # print(ft_clean)
        ft_response = ""
        ft_clean = ""
        result["models"]["fine-tuned"] = {
            "raw_response": ft_response,
            "clean_mermaid": ft_clean,
        }

        all_results.append(result)

    # Save raw responses
    with open(RAW_OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

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

    print(f"\nAll results saved to {RAW_OUTPUT_FILE} and {CLEAN_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
