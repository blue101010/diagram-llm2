import os
import asyncio
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from synthetic_dataset_generator.config import logger, MAX_WORKERS
from synthetic_dataset_generator.utils import append_to_output_file
from synthetic_dataset_generator.generators import generate_mermaid_diagram, generate_questions

async def process_question_batch(
    questions: List[str], diagram_type: str, doc_content: str
) -> List[Dict[str, str]]:
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

            logger.info(f"[process_question_batch] Generated diagram for: {question[:50]}...")

    return results


async def process_documentation_file(md_file: str, md_directory: str) -> List[Dict[str, str]]:
    """Process a single documentation file to generate questions and diagrams."""
    diagram_type = os.path.splitext(md_file)[0]
    md_path = os.path.join(md_directory, md_file)
    logger.info(
        f"\n[process_documentation_file] Processing documentation for diagram type: {diagram_type}"
    )

    try:
        # Input Validation: Check file size (limit to 5MB)
        file_size = os.path.getsize(md_path)
        if file_size > 5 * 1024 * 1024:  # 5MB
            logger.warning(f"[process_documentation_file] File '{md_path}' is too large ({file_size} bytes). Skipping.")
            return []
        
        if file_size == 0:
            logger.warning(f"[process_documentation_file] File '{md_path}' is empty. Skipping.")
            return []

        with open(md_path, "r", encoding="utf-8") as file:
            doc_content = file.read()
            
        # Input Validation: Check for empty content after read
        if not doc_content.strip():
             logger.warning(f"[process_documentation_file] File '{md_path}' contains only whitespace. Skipping.")
             return []
             
    except Exception as e:
        logger.error(f"[process_documentation_file] Error reading file '{md_path}': {e}", exc_info=True)
        return []

    # Step 1: Generate questions using Gemini 2.5 Pro
    questions = generate_questions(diagram_type, doc_content)
    if not questions:
        logger.warning(
            f"[process_documentation_file] Could not generate questions for '{diagram_type}'. Skipping."
        )
        return []

    logger.info(
        f"[process_documentation_file] Generated {len(questions)} questions for diagram type '{diagram_type}'."
    )

    # Step 2: Process questions in batches of 5 for parallel processing
    all_results = []
    batch_size = 5

    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        logger.info(
            f"[process_documentation_file] Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}"
        )

        batch_results = await process_question_batch(batch, diagram_type, doc_content)
        all_results.extend(batch_results)

        # Add a small delay between batches to avoid rate limiting
        await asyncio.sleep(1)

    return all_results
