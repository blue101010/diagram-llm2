# ============================================================================
# Inference & Integration Module: Use Fine-Tuned Qwen2.5-1.5B for Diagrams
# ============================================================================
# This module wraps the fine-tuned model for inference and integrates
# with diagram_llm2's generation pipeline.
# ============================================================================

import torch
import logging
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MermaidGenerationConfig:
    """Configuration for diagram generation."""
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1  # Set to >1 for beam search (slower on CPU)
    do_sample: bool = True
    repetition_penalty: float = 1.1


class DiagramGenerator:
    """
    Inference wrapper for fine-tuned Qwen2.5-1.5B diagram model.
    Handles prompt formatting, generation, and output validation.
    """
    
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        lora_adapter_path: str = None,
        device: str = "cpu",
    ):
        """
        Initialize the diagram generator.
        
        Args:
            base_model: Base model identifier (HF hub or local path)
            lora_adapter_path: Path to LoRA adapter weights
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.base_model = base_model
        self.lora_adapter_path = lora_adapter_path
        
        logger.info(f"Loading base model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device,
            torch_dtype=torch.float32,  # CPU: use float32
        )
        
        # Load LoRA adapter if provided
        if lora_adapter_path:
            logger.info(f"Loading LoRA adapter from {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_adapter_path,
                device_map=device,
            )
        
        self.model.eval()
        logger.info("Model loaded and set to eval mode")
    
    def generate_diagram(
        self,
        instruction: str,
        config: MermaidGenerationConfig = None,
    ) -> Dict[str, str]:
        """
        Generate a Mermaid diagram from a natural language instruction.
        
        Args:
            instruction: Natural language description of desired diagram
            config: Generation configuration parameters
            
        Returns:
            Dict with keys:
                - 'mermaid': Generated Mermaid diagram code
                - 'prompt': Full prompt sent to model
                - 'raw_output': Raw model output (with prompt)
        """
        if config is None:
            config = MermaidGenerationConfig()
        
        # Format prompt (same template as training)
        prompt = f"""Below is a request to generate a diagram in Mermaid syntax.

### Request:
{instruction}

### Mermaid Diagram:
"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            max_length=512,
            truncation=True,
        ).to(self.device)
        
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=config.max_length,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                num_beams=config.num_beams,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        raw_output = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        
        # Extract only the generated part (after prompt)
        mermaid_code = raw_output[len(prompt):].strip()
        
        return {
            "mermaid": mermaid_code,
            "prompt": prompt,
            "raw_output": raw_output,
        }
    
    def generate_multiple(
        self,
        instructions: list,
        config: MermaidGenerationConfig = None,
    ) -> list:
        """Generate diagrams for multiple instructions."""
        results = []
        for instruction in instructions:
            result = self.generate_diagram(instruction, config)
            results.append(result)
        return results
    
    def validate_mermaid(self, mermaid_code: str) -> tuple[bool, str]:
        """
        Basic validation of Mermaid syntax.
        For full validation, use: https://github.com/mermaid-js/mermaid
        
        Returns:
            (is_valid, error_message)
        """
        # Check for basic structure
        lines = mermaid_code.strip().split('\n')
        
        if not lines:
            return False, "Empty output"
        
        first_line = lines[0].lower()
        
        # Valid diagram types
        valid_types = [
            'graph', 'sequencediagram', 'flowchart', 'classDiagram',
            'stateDiagram', 'entityRelationshipDiagram', 'gantt',
            'pie', 'gitGraph', 'journey', 'timeline', 'quadrantChart',
        ]
        
        is_valid_type = any(
            t in first_line for t in valid_types
        )
        
        if not is_valid_type:
            return False, f"Invalid diagram type. Must start with one of: {valid_types}"
        
        # Basic bracket matching
        if mermaid_code.count('[') != mermaid_code.count(']'):
            return False, "Mismatched brackets [ ]"
        
        if mermaid_code.count('(') != mermaid_code.count(')'):
            return False, "Mismatched parentheses ( )"
        
        return True, ""


# ============================================================================
# INTEGRATION: Adapter for diagram_llm2's API
# ============================================================================

class Diagram_LLM2_Adapter:
    """
    Drop-in replacement for diagram_llm2's LLM backend.
    Adapts the fine-tuned local model to diagram_llm2's interface.
    """
    
    def __init__(self, lora_adapter_path: str = None):
        """Initialize the diagram generator."""
        self.generator = DiagramGenerator(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            lora_adapter_path=lora_adapter_path,
            device="cpu",
        )
        self.config = MermaidGenerationConfig(
            temperature=0.5,  # Lower temp for consistency
            top_p=0.95,
        )
    
    def generate(self, prompt: str) -> str:
        """
        Main generation method compatible with diagram_llm2's interface.
        
        diagram_llm2 expects: generate(prompt: str) -> str
        """
        result = self.generator.generate_diagram(prompt, self.config)
        return result["mermaid"]
    
    def generate_from_image(self, image_path: str) -> str:
        """
        (Placeholder) Diagram_llm2 supports image-to-diagram conversion.
        For now, this would require a vision model (e.g., Qwen2.5-VL).
        This script focuses on text-to-diagram.
        """
        raise NotImplementedError(
            "Image-to-diagram requires Qwen2.5-VL; "
            "see mermaid_finetune_cpu.py for text-to-diagram only"
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora-path",
        type=str,
        default="./lora_mermaid",
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Create a flowchart for a simple login process",
        help="Diagram generation instruction"
    )
    args = parser.parse_args()
    
    # Initialize generator
    logger.info("Initializing generator...")
    generator = DiagramGenerator(lora_adapter_path=args.lora_path)
    
    # Generate diagram
    logger.info(f"Generating diagram for: {args.instruction}")
    result = generator.generate_diagram(args.instruction)
    
    # Print results
    print("\n" + "="*60)
    print("GENERATED DIAGRAM:")
    print("="*60)
    print(result["mermaid"])
    
    # Validate
    is_valid, error = generator.validate_mermaid(result["mermaid"])
    print("\n" + "="*60)
    print(f"VALIDATION: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if error:
        print(f"Error: {error}")
    print("="*60)
