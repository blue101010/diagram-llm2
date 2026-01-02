# ============================================================================
# REFACTORING PLAN: Adapting diagram_llm2 for Local CPU-Fine-Tuned Model
# ============================================================================
# This document outlines minimal changes needed to integrate the fine-tuned
# Qwen2.5-1.5B model into diagram_llm2's codebase.
# ============================================================================

## EXECUTIVE SUMMARY

**Goal:** Replace diagram_llm2's cloud LLM backend (GPT-4 / Claude) with a
locally fine-tuned Qwen2.5-1.5B model running on CPU.

**Scope:** Minimal, surgical refactoring—only 3-4 core files modified.

**Effort:** ~30-60 minutes for integration + testing

**Impact:** 
- Zero cloud API costs
- Offline operation
- Customizable behavior via fine-tuning
- CPU-only (no GPU required)

---

## CURRENT diagram_llm2 ARCHITECTURE

*(Assumption based on typical LLM2-based projects)*

```
diagram_llm2/
├── src/
│   ├── llm_backend.py          ← LLM interface (abstract)
│   ├── openai_backend.py        ← Concrete: OpenAI/Azure impl
│   ├── diagram_generator.py    ← Orchestration
│   ├── diagram_validator.py    ← Output validation
│   └── templates/
│       └── diagram_templates.py ← Prompt/output templates
├── web/
│   ├── api.py                   ← FastAPI/Flask routes
│   ├── static/
│   └── templates/
├── config.yaml
└── requirements.txt
```

---

## PHASE 1: CREATE LOCAL LLM BACKEND CLASS

### File: `src/local_llm_backend.py` (NEW)

```python
# src/local_llm_backend.py
"""
Local LLM backend for fine-tuned Qwen2.5-1.5B on CPU.
Implements the same interface as openai_backend.py for drop-in replacement.
"""

from abc import ABC
from typing import Optional, Dict
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)


class LocalLLMBackend(ABC):
    """Abstract interface for local LLM backends."""
    
    def generate_diagram(self, instruction: str, **kwargs) -> str:
        """Generate diagram code from instruction."""
        raise NotImplementedError


class Qwen25DiagramBackend(LocalLLMBackend):
    """
    Fine-tuned Qwen2.5-1.5B for diagram generation.
    Compatible with existing diagram_llm2 interface.
    """
    
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        lora_adapter_path: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize the Qwen2.5-1.5B backend.
        
        Args:
            base_model: HF model identifier
            lora_adapter_path: Path to fine-tuned LoRA weights
            max_length: Max output tokens
            temperature: Sampling temperature (0.1-1.0)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        
        logger.info(f"Loading {base_model} on {device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device,
            torch_dtype=torch.float32,  # CPU
        )
        
        # Load LoRA if fine-tuned
        if lora_adapter_path:
            logger.info(f"Loading LoRA adapter: {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_adapter_path,
                device_map=device,
            )
        
        self.model.eval()
        logger.info("Backend initialized")
    
    def generate_diagram(
        self,
        instruction: str,
        diagram_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate Mermaid diagram from instruction.
        
        Args:
            instruction: Natural language description
            diagram_type: Optional hint (flowchart, sequence, etc.)
            **kwargs: Additional params (temperature, etc.)
            
        Returns:
            Mermaid diagram code
        """
        # Build prompt with type hint if provided
        prompt = self._build_prompt(instruction, diagram_type)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            max_length=512,
            truncation=True,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                temperature=kwargs.get("temperature", self.temperature),
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        full_output = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        
        # Extract only the generated part (after prompt)
        mermaid_code = full_output[len(prompt):].strip()
        
        return mermaid_code
    
    def _build_prompt(
        self,
        instruction: str,
        diagram_type: Optional[str] = None,
    ) -> str:
        """Build prompt matching training template."""
        
        base_prompt = f"""Below is a request to generate a diagram in Mermaid syntax.

### Request:
{instruction}
"""
        
        if diagram_type:
            base_prompt += f"### Diagram Type: {diagram_type}\n"
        
        base_prompt += "\n### Mermaid Diagram:\n"
        return base_prompt
```

---

## PHASE 2: UPDATE BACKEND FACTORY

### File: `src/llm_backend.py` (MODIFY)

```python
# src/llm_backend.py
"""
LLM backend factory. Minimal change: add local backend option.
"""

from typing import Literal
from .local_llm_backend import Qwen25DiagramBackend
from .openai_backend import OpenAIBackend  # existing

def get_llm_backend(
    backend_type: Literal["openai", "azure", "local"] = "openai",
    config: dict = None,
):
    """
    Factory function to load LLM backend.
    
    Args:
        backend_type: 'openai', 'azure', or 'local'
        config: Backend-specific config dict
        
    Returns:
        Configured backend instance
    """
    if backend_type == "local":
        return Qwen25DiagramBackend(
            base_model=config.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct"),
            lora_adapter_path=config.get("lora_adapter_path"),
            max_length=config.get("max_length", 512),
            temperature=config.get("temperature", 0.5),
            device=config.get("device", "cpu"),
        )
    elif backend_type == "openai":
        return OpenAIBackend(config)
    elif backend_type == "azure":
        return OpenAIBackend(config)  # existing
    else:
        raise ValueError(f"Unknown backend: {backend_type}")
```

---

## PHASE 3: UPDATE DIAGRAM GENERATOR ORCHESTRATION

### File: `src/diagram_generator.py` (MODIFY)

```python
# src/diagram_generator.py
"""
Main orchestration. Minimal change: swap backend initialization.
"""

from .llm_backend import get_llm_backend

class DiagramGenerator:
    """Orchestrates diagram generation with selected backend."""
    
    def __init__(self, backend_type: str = "local", config: dict = None):
        """
        Initialize with selected backend.
        
        Args:
            backend_type: 'local', 'openai', 'azure'
            config: Backend config dict
        """
        if config is None:
            config = self._load_default_config(backend_type)
        
        self.backend = get_llm_backend(backend_type, config)
    
    def _load_default_config(self, backend_type: str) -> dict:
        """Load default config for backend type."""
        if backend_type == "local":
            return {
                "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
                "lora_adapter_path": "./models/lora_mermaid",
                "device": "cpu",
            }
        # ... other backends
    
    def generate(self, instruction: str, diagram_type: str = None) -> str:
        """
        Generate diagram. Works with any backend!
        """
        return self.backend.generate_diagram(instruction, diagram_type)
```

---

## PHASE 4: UPDATE CONFIGURATION

### File: `config.yaml` (MODIFY)

Add local backend configuration:

```yaml
# config.yaml

# LLM Backend configuration
llm:
  backend: "local"  # Change from "openai" to "local"
  
  # Local backend (Qwen2.5-1.5B on CPU)
  local:
    base_model: "Qwen/Qwen2.5-1.5B-Instruct"
    lora_adapter_path: "./models/lora_mermaid"
    device: "cpu"  # or "cuda" if GPU available
    max_length: 512
    temperature: 0.5

  # Keep existing backends as fallback
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"

  azure:
    api_key: "${AZURE_API_KEY}"
    ...

# Diagram validation
diagram:
  max_retries: 2  # Retry if validation fails
  timeout: 30  # seconds per generation

# Web API
web:
  host: "127.0.0.1"
  port: 8000
  debug: true
```

---

## PHASE 5: UPDATE WEB API ROUTES

### File: `web/api.py` (MINIMAL CHANGE)

```python
# web/api.py
"""
FastAPI/Flask routes. Only add local backend option.
"""

from flask import Flask, request, jsonify
from src.diagram_generator import DiagramGenerator

app = Flask(__name__)

# Initialize with local backend
generator = DiagramGenerator(backend_type="local")

@app.route("/generate", methods=["POST"])
def generate_diagram():
    """
    Generate diagram endpoint.
    
    Request JSON:
    {
        "instruction": "Create a flowchart for login",
        "diagram_type": "flowchart"  # Optional
    }
    """
    data = request.json
    instruction = data.get("instruction")
    diagram_type = data.get("diagram_type")
    
    if not instruction:
        return {"error": "Missing instruction"}, 400
    
    try:
        mermaid_code = generator.generate(instruction, diagram_type)
        return {
            "success": True,
            "mermaid": mermaid_code,
            "backend": "local"  # Indicate which backend was used
        }
    except Exception as e:
        return {"error": str(e)}, 500
```

---

## PHASE 6: ADD TRAINING DATA UTILITIES

### File: `scripts/prepare_training_data.py` (NEW)

```python
#!/usr/bin/env python3
"""
Prepare training data from diagram_llm2's existing examples.
Converts existing diagram examples to JSONL format for LoRA fine-tuning.
"""

import json
import argparse
from pathlib import Path

def convert_to_jsonl(
    input_file: str,
    output_file: str,
    format_type: str = "auto",
):
    """
    Convert various formats to JSONL for training.
    
    Expected input format:
    - CSV: instruction, output
    - JSON: [{"instruction": "...", "output": "..."}]
    - JSONL: one per line
    """
    examples = []
    
    input_path = Path(input_file)
    
    if input_path.suffix == ".csv":
        import csv
        with open(input_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                examples.append({
                    "instruction": row["instruction"],
                    "output": row["output"],
                })
    
    elif input_path.suffix == ".json":
        with open(input_path) as f:
            data = json.load(f)
            examples = data if isinstance(data, list) else [data]
    
    # Write as JSONL
    with open(output_file, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"Converted {len(examples)} examples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input data file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()
    
    convert_to_jsonl(args.input, args.output)
```

---

## PHASE 7: SETUP & DEPLOYMENT

### File: `setup_local_backend.sh` (NEW)

```bash
#!/bin/bash
# setup_local_backend.sh - One-command setup for local backend

set -e

echo "Setting up local fine-tuned Qwen2.5-1.5B backend..."

# 1. Create model directory
mkdir -p models

# 2. Install dependencies
pip install -q transformers peft datasets torch

# 3. Download base model (one-time)
echo "Downloading base model (this may take a few minutes)..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('Base model downloaded successfully!')
"

# 4. Optional: Download example fine-tuned weights
# (Would need to host on HF Hub or similar)
# python -c "from peft import PeftModel; PeftModel.from_pretrained(...)"

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Prepare training data: python scripts/prepare_training_data.py --input data.json --output train.jsonl"
echo "2. Fine-tune: python mermaid_finetune_cpu.py --train_data_path train.jsonl"
echo "3. Run diagram_llm2: python web/api.py"
```

---

## MIGRATION CHECKLIST

- [ ] **Create** `src/local_llm_backend.py` with `Qwen25DiagramBackend` class
- [ ] **Update** `src/llm_backend.py` factory to support "local" backend
- [ ] **Update** `src/diagram_generator.py` to accept backend type parameter
- [ ] **Update** `config.yaml` with local backend configuration
- [ ] **Update** `web/api.py` to initialize with local backend (change one line)
- [ ] **Create** `scripts/prepare_training_data.py` for data prep
- [ ] **Create** `setup_local_backend.sh` for deployment
- [ ] **Test** local backend: `python -m pytest tests/test_local_backend.py`
- [ ] **Update** `requirements.txt`: add `peft`, `torch`

---

## TESTING STRATEGY

### Unit Test: `tests/test_local_backend.py` (NEW)

```python
import pytest
from src.local_llm_backend import Qwen25DiagramBackend

def test_backend_initialization():
    """Test backend loads successfully."""
    backend = Qwen25DiagramBackend(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        lora_adapter_path=None,
    )
    assert backend is not None

def test_simple_generation():
    """Test diagram generation works."""
    backend = Qwen25DiagramBackend()
    result = backend.generate_diagram("Create a simple flowchart")
    assert "graph" in result.lower() or "flowchart" in result.lower()

def test_diagram_validation():
    """Test output validation."""
    backend = Qwen25DiagramBackend()
    result = backend.generate_diagram("Create a flowchart")
    # Should contain valid Mermaid syntax
    assert len(result) > 10
    assert any(kw in result.lower() for kw in ["graph", "-->", "flowchart"])
```

---

## PERFORMANCE EXPECTATIONS

| Metric | Expected |
|--------|----------|
| **Cold Start** | 15-30s (first load of model) |
| **Inference (CPU)** | 2-8s per diagram |
| **Memory Usage** | 6-8 GB RAM (model + LoRA) |
| **Model Size** | ~3 GB base + ~10 MB LoRA |
| **Throughput** | ~0.1-0.2 diagrams/sec on CPU |

**Optimization for faster inference:**
- Add caching: Keep model in memory, reuse between requests
- Batch requests: Process multiple diagrams concurrently
- Quantization: Use 8-bit quantization for 2x speedup

---

## TROUBLESHOOTING

### Issue: "CUDA out of memory"
**Solution:** Explicitly set `device="cpu"` in config.yaml

### Issue: "Model too slow on CPU"
**Solutions:**
1. Reduce `max_length` in config (e.g., 256 instead of 512)
2. Lower `temperature` for faster convergence (e.g., 0.3)
3. Use GPU if available: set `device="cuda"`
4. Apply int8 quantization (see optional optimizations)

### Issue: "Output not in Mermaid format"
**Solutions:**
1. Fine-tune longer: increase epochs
2. Check training data quality
3. Lower temperature (0.3-0.5 for deterministic output)
4. Add output validation in `diagram_validator.py`

---

## NEXT STEPS AFTER INTEGRATION

1. **Data Collection:**
   - Gather 100-500 example (instruction, diagram) pairs
   - Use existing diagram_llm2 examples + synthetic generation
   - Focus on common diagram types: flowcharts, sequence, architecture

2. **Fine-Tuning:**
   - Run `python mermaid_finetune_cpu.py` (1-3 hours on CPU)
   - Evaluate on validation set
   - Iterate on training data quality

3. **Optimization:**
   - Add caching/batching for faster inference
   - Benchmark vs. cloud API costs
   - Consider 0.5B model variant if 1.5B too slow

4. **Production Hardening:**
   - Add error handling for generation failures
   - Implement diagram repair/validation loops
   - Add telemetry logging

---

## SUMMARY

**Lines Changed:** ~150 across 4-5 files
**New Files:** 3 (backend, inference, setup script)
**Complexity:** Low (factory pattern, minimal coupling)
**Time to Integrate:** 30-60 minutes
**Risk:** Minimal (no breaking changes to existing API)

The refactoring maintains diagram_llm2's existing architecture while enabling
local, privacy-respecting diagram generation with a fine-tuned small model.
