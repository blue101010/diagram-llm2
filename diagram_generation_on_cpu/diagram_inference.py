# ============================================================================
# Inference & Integration Module: Use Fine-Tuned Qwen2.5-1.5B for Diagrams
# ============================================================================
# This module wraps the fine-tuned model for inference and integrates
# with diagram_llm2's generation pipeline.
# ============================================================================

import logging
import re
import subprocess
import tempfile
import shutil
import os
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

# Heavy libraries (torch, transformers, peft) are imported lazily inside the class
# to prevent slow startup when just importing the module.


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
        lora_adapter_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize the diagram generator.
        
        Args:
            base_model: Base model identifier (HF hub or local path)
            lora_adapter_path: Path to LoRA adapter weights
            device: 'cpu' or 'cuda'
        """
        # Lazy import of heavy libraries
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self.device = device
        self.base_model = base_model
        self.lora_adapter_path = lora_adapter_path
        
        logger.info(f"Loading base model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device,
            dtype=torch.float32,  # CPU: use float32
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
        config: Optional[MermaidGenerationConfig] = None,
        stream_output: bool = False,
    ) -> Dict[str, str]:
        """
        Generate a Mermaid diagram from a natural language instruction.
        
        Args:
            instruction: Natural language description of desired diagram
            config: Generation configuration parameters
            stream_output: If True, stream generated text to stdout
            
        Returns:
            Dict with keys:
                - 'mermaid': Generated Mermaid diagram code
                - 'prompt': Full prompt sent to model
                - 'raw_output': Raw model output (with prompt)
        """
        import torch  # Lazy import
        from transformers import TextStreamer

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
        
        # Setup streamer if requested
        streamer = None
        if stream_output:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)

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
                streamer=streamer,
            )
        
        # Decode output
        raw_output = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        
        # Extract only the generated part (after prompt)
        raw_generated_text = raw_output[len(prompt):].strip()
        
        # Extract clean Mermaid code
        mermaid_code = self._extract_mermaid_code(raw_generated_text)
        
        return {
            "mermaid": mermaid_code,
            "prompt": prompt,
            "raw_output": raw_output,
        }
    
    def _extract_mermaid_code(self, text: str) -> str:
        """Extract Mermaid code from text, handling markdown blocks."""
        # Pattern 1: Markdown code block with mermaid identifier
        pattern_mermaid = r"```mermaid\s*(.*?)\s*```"
        match = re.search(pattern_mermaid, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
        # Pattern 2: Generic markdown code block
        pattern_generic = r"```\s*(.*?)\s*```"
        match = re.search(pattern_generic, text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # Pattern 3: No code blocks, just return text
        # But try to strip leading non-code text if possible
        return text.strip()

    def generate_multiple(
        self,
        instructions: list,
        config: Optional[MermaidGenerationConfig] = None,
    ) -> list:
        """Generate diagrams for multiple instructions."""
        results = []
        for instruction in instructions:
            result = self.generate_diagram(instruction, config)
            results.append(result)
        return results
    
    def validate_mermaid(self, mermaid_code: str) -> tuple[bool, str]:
        """
        Validate Mermaid syntax using the official Mermaid CLI (mmdc) if available.
        Falls back to basic validation if mmdc is not found.
        
        To enable full validation: npm install -g @mermaid-js/mermaid-cli
        
        Returns:
            (is_valid, error_message)
        """
        # 1. Try Official Mermaid CLI (mmdc)
        mmdc_path = None
        
        # On Windows, check specific paths
        if os.name == 'nt':
            # 1. Check PATH for mmdc.cmd
            mmdc_path = shutil.which("mmdc.cmd")
            
            # 2. Check APPDATA/npm (common install location)
            if not mmdc_path:
                appdata = os.getenv('APPDATA')
                if appdata:
                    possible_path = os.path.join(appdata, 'npm', 'mmdc.cmd')
                    if os.path.exists(possible_path):
                        mmdc_path = possible_path
        
        # Fallback or non-Windows
        if not mmdc_path:
            mmdc_path = shutil.which("mmdc")

        if mmdc_path:
            logger.debug(f"Attempting validation with Mermaid CLI at: {mmdc_path}")
            try:
                # Create temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False, encoding='utf-8') as tmp:
                    tmp.write(mermaid_code)
                    tmp_path = tmp.name
                
                # Output to a temp file to avoid clutter
                out_svg = tmp_path + ".svg"
                
                # Run mmdc
                # We use --quiet to reduce noise, but capture stderr for errors
                # On Windows, if using .cmd, shell=True might be safer, but full path usually works.
                cmd = [mmdc_path, "-i", tmp_path, "-o", out_svg]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=False,
                    shell=(os.name == 'nt') # Use shell=True on Windows to ensure .cmd execution
                )
                
                # Cleanup
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    if os.path.exists(out_svg):
                        os.unlink(out_svg)
                except OSError:
                    pass # Ignore cleanup errors
                
                if result.returncode == 0:
                    return True, ""
                else:
                    # Extract error from stderr
                    return False, f"Mermaid CLI Error: {result.stderr.strip()}"
                    
            except Exception as e:
                logger.warning(f"Mermaid CLI validation failed to run: {e}. Falling back to basic validation.")
        else:
            # Only log once per session ideally, but here we log every time it's missing if we want to be annoying, 
            # or just debug. Let's use debug.
            logger.warning("Mermaid CLI (mmdc) not found. Using basic validation.")

        # 2. Basic Validation (Fallback)
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
            'erDiagram', 'mindmap', 'zenuml', 'sankey-beta', 'xychart-beta', 'block-beta', 'packet-beta'
        ]
        
        is_valid_type = any(
            t.lower() in first_line for t in valid_types
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
    
    def __init__(self, lora_adapter_path: Optional[str] = None):
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
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Single instruction to test"
    )
    parser.add_argument(
        "--validation-file",
        type=str,
        default="../gemini_fine_tune/dataset/validation_data.jsonl",
        help="Path to validation file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of examples to process"
    )
    args = parser.parse_args()
    
    # Initialize generator
    logger.info("Initializing generator...")
    # Use Qwen2.5-1.5B-Instruct as base
    generator = DiagramGenerator(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        lora_adapter_path=args.lora_path
    )
    
    instructions = []
    if args.instruction:
        instructions = [args.instruction]
    else:
        # Load from validation file
        if Path(args.validation_file).exists():
            logger.info(f"Loading examples from {args.validation_file}")
            with open(args.validation_file, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    if count >= args.limit: break
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if "contents" in item:
                                instructions.append(item["contents"][0]["parts"][0]["text"])
                                count += 1
                        except Exception:
                            continue
        else:
            logger.warning(f"Validation file {args.validation_file} not found. Using default.")
            instructions = ["Create a flowchart for a simple login process"]

    for instr in instructions:
        logger.info(f"Generating diagram for: {instr[:50]}...")
        result = generator.generate_diagram(instr)
        
        print("\n" + "="*60)
        print(f"INSTRUCTION: {instr}")
        print("-" * 60)
        print("GENERATED DIAGRAM:")
        print("-" * 60)
        print(result["mermaid"])
        
        # Validate
        is_valid, error = generator.validate_mermaid(result["mermaid"])
        print("\n" + "="*60)
        print(f"VALIDATION: {'✓ PASS' if is_valid else '✗ FAIL'}")
        if error:
            print(f"Error: {error}")
        print("="*60)
