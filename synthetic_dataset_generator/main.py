import os
import json
import asyncio
from synthetic_dataset_generator import config
from synthetic_dataset_generator.config import logger, OUTPUT_FILE, MD_DIRECTORY, DEFAULT_MODEL
from synthetic_dataset_generator.processor import process_documentation_file
from synthetic_dataset_generator.utils import select_model, load_model_limits

async def main() -> None:
    """Main function to process all documentation files."""
    # Interactive Model Selection
    model_id = select_model(DEFAULT_MODEL)
    
    # Update Rate Limit based on selection
    limits = load_model_limits()
    if model_id in limits:
        rpm = limits[model_id].get("rpm", 0)
        if rpm > 0:
            # Calculate delay: 60 / (RPM - 1) to be safe
            target_rpm = max(1, rpm - 1)
            new_delay = 60.0 / target_rpm
            config.RATE_LIMIT_DELAY = new_delay
            logger.info(f"[main] Selected model: {model_id}")
            logger.info(f"[main] Updated RATE_LIMIT_DELAY to {new_delay:.2f}s (Target RPM: {target_rpm})")
    
    dataset = []

    # Initialize the output file if it doesn't exist
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        logger.info(f"[main] Initialized output file at {OUTPUT_FILE}")

    if not os.path.exists(MD_DIRECTORY):
        logger.error(
            f"[main] The directory '{MD_DIRECTORY}' does not exist. Please create it and add your markdown documentation files."
        )
        return

    try:
        md_files = [f for f in os.listdir(MD_DIRECTORY) if f.endswith(".md")]
    except Exception as e:
        logger.error(f"[main] Error reading directory '{MD_DIRECTORY}': {e}", exc_info=True)
        return

    if not md_files:
        logger.warning(f"[main] No markdown files found in '{MD_DIRECTORY}'.")
        return

    logger.info(
        f"[main] Starting mermaid diagram generation. Results will be saved to {OUTPUT_FILE}"
    )
    logger.info(
        f"[main] To view results in a web browser, run 'python webapp_server.py' in a separate terminal"
    )

    for md_file in md_files:
        results = await process_documentation_file(md_file, MD_DIRECTORY, model_id)
        dataset.extend(results)

    logger.info("\n[main] Generation complete.")
    logger.info(f"[main] Generated a total of {len(dataset)} entries.")
    logger.info(f"[main] All results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n[main] Keyboard interrupt received. Shutting down.")
    except Exception as e:
        logger.critical(f"[main] Error in main function: {e}", exc_info=True)
