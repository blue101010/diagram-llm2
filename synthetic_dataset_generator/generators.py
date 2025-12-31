import json
import os
from typing import List
from google import genai
from google.genai import types

from synthetic_dataset_generator.config import (
    logger,
    GEMINI_2_5_PRO,
    QUESTION_PROMPT,
    MERMAID_PROMPT
)
from synthetic_dataset_generator.utils import call_gemini_with_retry

def generate_questions(diagram_type: str, doc_content: str) -> List[str]:
    """Generate questions for the given diagram type using Gemini 2.5 Pro."""
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        prompt = QUESTION_PROMPT.format(
            diagram_type=diagram_type, doc_content=doc_content
        )

        response = call_gemini_with_retry(
            client,
            GEMINI_2_5_PRO,
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "questions": {"type": "ARRAY", "items": {"type": "STRING"}}
                    },
                    "required": ["questions"],
                },
            ),
        )

        if not response:
            logger.warning(
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
                logger.info(
                    f"[generate_questions] Successfully generated {len(questions)} questions"
                )
                return questions
            else:
                logger.warning(
                    f"[generate_questions] Response missing questions array: {response.text}"
                )
                return []
        except json.JSONDecodeError as e:
            logger.error(f"[generate_questions] Error parsing response JSON: {e}")
            logger.debug(f"Response text: {response.text}")
            return []

    except Exception as e:
        logger.error(f"[generate_questions] Error generating questions with Gemini: {e}", exc_info=True)
        return []


def generate_mermaid_diagram(question: str, diagram_type: str, doc_content: str) -> str:
    """Generate a Mermaid diagram using Gemini 2.5 Pro."""
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        prompt = MERMAID_PROMPT.format(
            diagram_type=diagram_type, doc_content=doc_content, question=question
        )

        response = call_gemini_with_retry(
            client,
            GEMINI_2_5_PRO,
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {"mermaid_diagram": {"type": "STRING"}},
                    "required": ["mermaid_diagram"],
                },
            ),
        )

        if not response:
            logger.warning(
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
                logger.warning(
                    f"[generate_mermaid_diagram] Response missing mermaid_diagram: {response.text}"
                )
                return ""
        except json.JSONDecodeError as e:
            logger.error(f"[generate_mermaid_diagram] Error parsing response JSON: {e}")
            logger.debug(f"Response text: {response.text}")
            return ""

    except Exception as e:
        logger.error(
            f"[generate_mermaid_diagram] Error generating Mermaid diagram with Gemini: {e}", exc_info=True
        )
        return ""
