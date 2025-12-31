import os
import json
import asyncio
import traceback
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, List, Dict, Optional

import google.generativeai as genai
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------ CONFIGURATION & CONSTANTS ------------------

# Model identifiers
GEMINI_2_5_PRO = "gemini-2.5-pro-preview-03-25"

# API Rate Limiting
MAX_WORKERS = 3  
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # Seconds

# File to save all generated questions and diagrams
OUTPUT_FILE = "generated_questions.json"

# Multi-line string constants for prompts
QUESTION_PROMPT = """\
You are an expert question generator for Mermaid diagrams.
Generate 100 diverse questions that require a Mermaid diagram as an answer.
The Mermaid diagram should be of type "{diagram_type}".
All generated questions must have multiple interacting components. They should not be about a single component.
They should be relatively complex and also make sure there are many complex questions with more than 5 components and many arrows and interactions.

The questions should focus on teaching proper Mermaid syntax rather than complex concepts.
Each question should specify EXACTLY what elements should be in the diagram.

For example:
- "Draw a pie chart with sections A (20%), B (70%), and C (10%)"
- "Draw a sequence diagram where User logs in to System, System validates credentials, and then System grants access"
- "Draw a flow diagram where A sends request to B, and B responds to the request with a delay of 2ms"

Include questions of varying complexity, covering ALL syntax features described in the documentation.
Your questions should systematically cover every element, feature, and syntax option in the documentation.

Here is the complete documentation for this diagram type that you should use to generate questions:
```
{doc_content}
```

Return your answer strictly as a JSON object with ONE key "questions" containing an array of 100 strings.
Do not include any additional text.

MAKE SURE THE QUESTION SPECIFIES EXACTLY WHAT SHOULD BE IN THE DIAGRAM. THERE SHOULD BE NO AMBIGUITY.
I don't want to see any questions with less than 3 components.
"""

MERMAID_PROMPT = """\
You are a Mermaid diagram generator. Below is the documentation for Mermaid diagrams of type "{diagram_type}":

## Documentation
```markdown
{doc_content}
```

## Question:
{question}

## Instructions
Based on the documentation above, generate a precise and correct Mermaid diagram that answers the question.
Return your answer strictly as a JSON object with ONE key "mermaid_diagram" whose value is a valid Mermaid code snippet.
Do not include any additional explanation or text - ONLY the Mermaid code.

The Mermaid code MUST be syntactically correct and render without errors according to the provided Mermaid documentation.
- Pay close attention to syntax rules, keywords, and allowed characters.
- Ensure the diagram properly represents what was asked in the question.
- THE OUTPUT DIAGRAM MUST LOOK PRETTY and properly spaced. It shouldn't be cluttered.
- Make sure when you're drawing arrows or writing large texts they aren't overlapping.
- Only give the mermaid code, no text not even markdown code wrapper. Just the mermaid code directly!

Eg - Create an architecture diagram with a group `cloud_env` (cloud icon, title 'Cloud Environment'). Inside this group, place two services: `app` (server icon, 'Application') and `storage` (disk icon, 'File Storage'). Connect `app`'s right port (`R`) to `storage`'s left port (`L`).
In the output the mermaid_diagram key should have the following value:
architecture-beta
    group cloud_env(cloud)[Cloud Environment]

    service app(server)[Application] in cloud_env
    service storage(disk)[File Storage] in cloud_env

    app:R -- L:storage
"""


# ------------------ STRUCTURED OUTPUT MODELS ------------------


class QuestionsOutput(BaseModel):
    questions: List[str]


class MermaidDiagramOutput(BaseModel):
    mermaid_diagram: str


# ------------------ HELPER FUNCTIONS ------------------


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
                print(f"[safe_json_loads] Regex extraction failed: {e}")
                traceback.print_exc()
                raise
        else:
            raise


def call_gemini_with_retry(model, prompt, generation_config):
    """Call Gemini API with exponential backoff retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            return model.generate_content(prompt, generation_config=generation_config)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"[call_gemini_with_retry] Failed after {MAX_RETRIES} attempts: {e}")
                raise e
            
            delay = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
            print(f"[call_gemini_with_retry] API Error: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)


def generate_questions(diagram_type: str, doc_content: str) -> List[str]:
    """Generate questions for the given diagram type using Gemini 2.5 Pro."""
    try:
        model = genai.GenerativeModel(GEMINI_2_5_PRO)
        prompt = QUESTION_PROMPT.format(
            diagram_type=diagram_type, doc_content=doc_content
        )

        response = call_gemini_with_retry(
            model,
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {
                        "questions": {"type": "ARRAY", "items": {"type": "STRING"}}
                    },
                    "required": ["questions"],
                },
            },
        )

        if not response:
            print(
                f"[generate_questions] Gemini did not return any response for diagram type: {diagram_type}"
            )
            return []

        # Parse the JSON output from the text response
        try:
            json_output = json.loads(response.text)
            if "questions" in json_output and isinstance(
                json_output["questions"], list
            ):
                questions = json_output["questions"]
                print(
                    f"[generate_questions] Successfully generated {len(questions)} questions"
                )
                return questions
            else:
                print(
                    f"[generate_questions] Response missing questions array: {response.text}"
                )
                return []
        except json.JSONDecodeError as e:
            print(f"[generate_questions] Error parsing response JSON: {e}")
            print(f"Response text: {response.text}")
            return []

    except Exception as e:
        print(f"[generate_questions] Error generating questions with Gemini: {e}")
        traceback.print_exc()
        return []


def generate_mermaid_diagram(question: str, diagram_type: str, doc_content: str) -> str:
    """Generate a Mermaid diagram using Gemini 2.5 Pro."""
    try:
        model = genai.GenerativeModel(GEMINI_2_5_PRO)
        prompt = MERMAID_PROMPT.format(
            diagram_type=diagram_type, doc_content=doc_content, question=question
        )

        response = call_gemini_with_retry(
            model,
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {"mermaid_diagram": {"type": "STRING"}},
                    "required": ["mermaid_diagram"],
                },
            },
        )

        if not response:
            print(
                f"[generate_mermaid_diagram] Gemini did not return any response for question: {question}"
            )
            return ""

        # Parse the JSON output from the text response
        try:
            json_output = json.loads(response.text)
            if "mermaid_diagram" in json_output:
                mermaid_code = json_output["mermaid_diagram"].strip()
                return mermaid_code
            else:
                print(
                    f"[generate_mermaid_diagram] Response missing mermaid_diagram: {response.text}"
                )
                return ""
        except json.JSONDecodeError as e:
            print(f"[generate_mermaid_diagram] Error parsing response JSON: {e}")
            print(f"Response text: {response.text}")
            return ""

    except Exception as e:
        print(
            f"[generate_mermaid_diagram] Error generating Mermaid diagram with Gemini: {e}"
        )
        traceback.print_exc()
        return ""


async def process_question_batch(
    questions: List[str], diagram_type: str, doc_content: str
) -> List[Dict]:
    """Process a batch of questions in parallel using ThreadPoolExecutor."""
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create partial function with fixed arguments
        generate_diagram_for_question = partial(
            generate_mermaid_diagram, diagram_type=diagram_type, doc_content=doc_content
        )

        # Create a list of futures
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(executor, generate_diagram_for_question, question)
            for question in questions
        ]

        # Wait for all futures to complete
        for question, future in zip(questions, futures):
            mermaid_diagram = await future

            # Create entry
            entry = {
                "diagram_type": diagram_type,
                "prompt": question,
                "output": mermaid_diagram,
            }

            results.append(entry)

            # Append to the output file immediately
            append_to_output_file(entry)

            print(f"[process_question_batch] Generated diagram for: {question[:50]}...")

    return results


def append_to_output_file(entry: Dict):
    """Append a single entry to the output file."""
    try:
        # Read existing data
        existing_data = []
        if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"[append_to_output_file] Error parsing {OUTPUT_FILE}, starting fresh"
                    )
                    existing_data = []

        # Append new entry
        existing_data.append(entry)

        # Write back to file
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)

    except Exception as e:
        print(f"[append_to_output_file] Error updating output file: {e}")
        traceback.print_exc()


async def process_documentation_file(md_file: str, md_directory: str) -> List[Dict]:
    """Process a single documentation file to generate questions and diagrams."""
    diagram_type = os.path.splitext(md_file)[0]
    md_path = os.path.join(md_directory, md_file)
    print(
        f"\n[process_documentation_file] Processing documentation for diagram type: {diagram_type}"
    )

    try:
        # Input Validation: Check file size (limit to 5MB)
        file_size = os.path.getsize(md_path)
        if file_size > 5 * 1024 * 1024:  # 5MB
            print(f"[process_documentation_file] File '{md_path}' is too large ({file_size} bytes). Skipping.")
            return []
        
        if file_size == 0:
            print(f"[process_documentation_file] File '{md_path}' is empty. Skipping.")
            return []

        with open(md_path, "r", encoding="utf-8") as file:
            doc_content = file.read()
            
        # Input Validation: Check for empty content after read
        if not doc_content.strip():
             print(f"[process_documentation_file] File '{md_path}' contains only whitespace. Skipping.")
             return []
             
    except Exception as e:
        print(f"[process_documentation_file] Error reading file '{md_path}': {e}")
        traceback.print_exc()
        return []

    # Step 1: Generate questions using Gemini 2.5 Pro
    questions = generate_questions(diagram_type, doc_content)
    if not questions:
        print(
            f"[process_documentation_file] Could not generate questions for '{diagram_type}'. Skipping."
        )
        return []

    print(
        f"[process_documentation_file] Generated {len(questions)} questions for diagram type '{diagram_type}'."
    )

    # Step 2: Process questions in batches of 5 for parallel processing
    all_results = []
    batch_size = 5

    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        print(
            f"[process_documentation_file] Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}"
        )

        batch_results = await process_question_batch(batch, diagram_type, doc_content)
        all_results.extend(batch_results)

        # Add a small delay between batches to avoid rate limiting
        await asyncio.sleep(1)

    return all_results


# ------------------ MAIN FUNCTION ------------------


async def main():
    """Main function to process all documentation files."""
    dataset = []
    md_directory = "md"

    # Initialize the output file if it doesn't exist
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        print(f"[main] Initialized output file at {OUTPUT_FILE}")

    if not os.path.exists(md_directory):
        print(
            f"[main] The directory '{md_directory}' does not exist. Please create it and add your markdown documentation files."
        )
        return

    try:
        md_files = [f for f in os.listdir(md_directory) if f.endswith(".md")]
    except Exception as e:
        print(f"[main] Error reading directory '{md_directory}': {e}")
        traceback.print_exc()
        return

    if not md_files:
        print(f"[main] No markdown files found in '{md_directory}'.")
        return

    print(
        f"[main] Starting mermaid diagram generation. Results will be saved to {OUTPUT_FILE}"
    )
    print(
        f"[main] To view results in a web browser, run 'python webapp_server.py' in a separate terminal"
    )

    for md_file in md_files:
        results = await process_documentation_file(md_file, md_directory)
        dataset.extend(results)

    print("\n[main] Generation complete.")
    print(f"[main] Generated a total of {len(dataset)} entries.")
    print(f"[main] All results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[main] Keyboard interrupt received. Shutting down.")
    except Exception as e:
        print(f"[main] Error in main function: {e}")
        traceback.print_exc()
