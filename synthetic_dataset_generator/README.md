# Synthetic Dataset Generator

This subdirectory contains a tool for generating synthetic datasets of Mermaid.js diagrams using the Google Gemini API. It reads documentation from markdown files and generates pairs of "questions" (prompts) and corresponding Mermaid diagram code.

## Purpose

The goal is to create a large dataset of valid Mermaid diagrams to train or fine-tune models, or to test diagram generation capabilities. It uses a two-step process:
1.  **Question Generation**: Generates diverse prompts/questions based on the provided documentation.
2.  **Diagram Generation**: Generates valid Mermaid code that answers those specific questions.

## Usage

1.  **Prerequisites**:
    *   Install dependencies (see root `requirements.txt` or `requirements.txt` in this folder if present).
    *   Set the `GOOGLE_API_KEY` environment variable with your Gemini API key.

2.  **Input**:
    *   Place markdown (`.md`) files containing Mermaid documentation in the `md/` directory.
    *   The filename (e.g., `sequence.md`) will be used as the `diagram_type` (e.g., `sequence`).

3.  **Run**:
    From the root of the repository:
    ```bash
    python -m synthetic_dataset_generator.main
    ```

4.  **Output**:
    *   Results are saved to `generated_questions.json`.
    *   Logs are written to `dataset_generation.log`.

## File Structure

*   **`main.py`**: The entry point of the application. It orchestrates the process: reading files from `md/`, calling the processor, and initializing the output file.
*   **`processor.py`**: Handles the core logic for each documentation file. It coordinates with generators to produce questions and then diagrams in parallel batches.
*   **`generators.py`**: Contains the logic for interacting with the Gemini API.
    *   `generate_questions`: Asks the model to create prompts based on documentation.
    *   `generate_mermaid_diagram`: Asks the model to generate Mermaid code for a specific prompt.
*   **`config.py`**: Configuration settings, including:
    *   Model names (e.g., `gemini-2.5-flash`).
    *   Prompts (`QUESTION_PROMPT`, `MERMAID_PROMPT`).
    *   File paths (`md/`, `generated_questions.json`).
    *   Concurrency settings (`MAX_WORKERS`).
*   **`utils.py`**: Utility functions for:
    *   Robust JSON parsing.
    *   API calls with exponential backoff and retries.
    *   Thread-safe file appending.
*   **`md/`**: Directory for input markdown documentation files.
*   **`generated_questions.json`**: The output file containing the generated dataset.
