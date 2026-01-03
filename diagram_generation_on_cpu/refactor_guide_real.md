#  Adapting diagram_llm2 for Local CPU-Fine-Tuned Model

## Goal

REFACTORING PLAN: Adapting diagram_llm2 for Local CPU-Fine-Tuned Model
This document outlines minimal changes needed to integrate the fine-tuned
Qwen2.5-1.5B model into diagram_llm2's codebase.

## EXECUTIVE SUMMARY

**Goal:** Replace diagram_llm2's cloud LLM backend (Google GenAI) with a
locally fine-tuned Qwen2.5-1.5B model running on CPU.

**Scope:** Minimal, surgical refactoring.

**Effort:** ~30-60 minutes for integration + testing

**Impact:** 
- Zero cloud API costs
- Offline operation
- Customizable behavior via fine-tuning
- CPU-only (no GPU required)

---

## CURRENT diagram_llm2 ARCHITECTURE

```
diagram_llm2/
├── gemini_fine_tune/
│   ├── perform_inference.py    ← Current inference script (Google GenAI)
│   └── dataset/                ← Training/Validation data
├── synthetic_dataset_generator/
│   └── generators.py           ← Data generation logic
├── diagram_generation_on_cpu/  ← NEW: Local CPU backend
│   ├── diagram_inference.py    ← Qwen2.5 inference wrapper
│   └── mermaid_finetune_cpu.py ← Training script
└── requirements.txt
```

---

## PHASE 1: INTEGRATE LOCAL BACKEND

### File: `gemini_fine_tune/local_inference.py` (NEW)

Create a new script that mimics `perform_inference.py` but uses the local model.

```python
# gemini_fine_tune/local_inference.py
import sys
import os
import json
import logging

# Add parent directory to path to import from diagram_generation_on_cpu
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from diagram_generation_on_cpu.diagram_inference import DiagramGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize local generator
    # Use Qwen2.5-1.5B-Instruct as base
    generator = DiagramGenerator(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        lora_adapter_path=None # Set to "./lora_mermaid_output" after training
    )
    
    # Load validation data
    validation_file = os.path.join(os.path.dirname(__file__), "dataset", "validation_data.jsonl")
    
    if os.path.exists(validation_file):
        logger.info(f"Loading examples from {validation_file}")
        with open(validation_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5: break # Limit to 5 examples
                if line.strip():
                    try:
                        item = json.loads(line)
                        if "contents" in item:
                            prompt = item["contents"][0]["parts"][0]["text"]
                            logger.info(f"Generating diagram for: {prompt[:50]}...")
                            result = generator.generate_diagram(prompt)
                            print("\n" + "="*60)
                            print(result["mermaid"])
                            print("="*60)
                    except Exception as e:
                        logger.error(f"Error processing line: {e}")
    else:
        logger.warning("Validation file not found.")

if __name__ == "__main__":
    main()
```

---

## PHASE 2: FINE-TUNE ON CPU

1.  **Install dependencies**:
    ```bash
    pip install -r diagram_generation_on_cpu/requirements.txt
    ```

2.  **Run Training**:
    ```bash
    cd diagram_generation_on_cpu
    python mermaid_finetune_cpu.py \
        --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
        --train_data_path ../gemini_fine_tune/dataset/training_data.jsonl \
        --output_dir ./lora_mermaid_output \
        --num_train_epochs 3
    ```

---

## PHASE 3: SWAP BACKEND IN APP

If you have a web app or other scripts using `perform_inference.py`, you can now replace the calls to Google GenAI with calls to `DiagramGenerator`.

For example, in `synthetic_dataset_generator/generators.py`, you could import `DiagramGenerator` and use it to generate synthetic examples locally.
