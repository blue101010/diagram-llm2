import json
import os
from typing import List
from google import genai
from google.genai import types

from synthetic_dataset_generator.config import (
    logger,
    QUESTION_PROMPT,
    MERMAID_PROMPT
)
from synthetic_dataset_generator.utils import call_gemini_with_retry, safe_json_loads

def generate_questions(diagram_type: str, doc_content: str, model_id: str) -> List[str]:
    """Generate questions for the given diagram type using the selected model."""
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        prompt = QUESTION_PROMPT.format(
            diagram_type=diagram_type, doc_content=doc_content
        )

        config = None
        # Only use JSON mode for Gemini models (checking loosely for "gemini-2" or "gemini-1.5")
        # Gemma models (e.g. gemma-3-27b-it) do not support response_mime_type="application/json"
        if "gemini" in model_id.lower() and "gemma" not in model_id.lower():
             config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "questions": {"type": "ARRAY", "items": {"type": "STRING"}}
                    },
                    "required": ["questions"],
                },
            )
        
        response = call_gemini_with_retry(
            client,
            model_id,
            prompt,
            config=config,
        )

        if not response:
            logger.warning(
                f"[generate_questions] Gemini did not return any response for diagram type: {diagram_type}"
            )
            return []

        # Parse the JSON output from the text response
        try:
            # use safe_json_loads because model might return markdown code block if json mode is off
            json_output = safe_json_loads(response.text)
            
            if "questions" in json_output and isinstance(
                json_output["questions"], list
            ):
                questions = json_output["questions"]
                logger.info(
                    f"[generate_questions] Successfully generated {len(questions)} questions"
                )
                return questions
            else:
                logger.warning(
                    f"[generate_questions] Response missing questions array: {response.text}"
                )
                return []
        except Exception as e:
            logger.error(f"[generate_questions] Error parsing response JSON: {e}")
            logger.debug(f"Response text: {response.text}")
            return []

    except Exception as e:
        logger.error(f"[generate_questions] Error generating questions with Gemini: {e}", exc_info=True)
        return []


def generate_mermaid_diagram(question: str, diagram_type: str, doc_content: str, model_id: str) -> str:
    """Generate a Mermaid diagram using the selected model."""
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        prompt = MERMAID_PROMPT.format(
            diagram_type=diagram_type, doc_content=doc_content, question=question
        )

        config = None
        # Only use JSON mode for Gemini models
        if "gemini" in model_id.lower() and "gemma" not in model_id.lower():
             config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {"mermaid_diagram": {"type": "STRING"}},
                    "required": ["mermaid_diagram"],
                },
            )

        response = call_gemini_with_retry(
            client,
            model_id,
            prompt,
            config=config,
        )

        if not response:
            logger.warning(
                f"[generate_mermaid_diagram] Gemini did not return any response for question: {question}"
            )
            return ""

        # Parse the JSON output from the text response
        try:
            json_output = safe_json_loads(response.text)
            if "mermaid_diagram" in json_output:
                mermaid_code = json_output["mermaid_diagram"].strip()
                return mermaid_code
            else:
                 # Fallback: if json parsing worked but key missing, maybe it returned direct code?
                 # But safer to log warning for now as prompt demanded JSON.
                logger.warning(
                    f"[generate_mermaid_diagram] Response missing mermaid_diagram: {response.text}"
                )
                return ""
        except Exception as e:
            logger.error(f"[generate_mermaid_diagram] Error parsing response JSON: {e}")
            logger.debug(f"Response text: {response.text}")
            return ""

    except Exception as e:
        logger.error(
            f"[generate_mermaid_diagram] Error generating Mermaid diagram with Gemini: {e}", exc_info=True
        )
        return ""
