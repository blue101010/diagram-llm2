# test_zero_shot_mermaid.py
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import diagram_inference
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from diagram_inference import DiagramGenerator
except ImportError as e:
    logger.error("Could not import diagram_inference.")
    logger.error(f"Error details: {e}")
    logger.error("Make sure you have installed the requirements:")
    logger.error("pip install -r ../requirements.txt")
    sys.exit(1)

def main():
    logger.info("Initializing DiagramGenerator (Zero-Shot)...")
    # Initialize with base model only (no LoRA)
    generator = DiagramGenerator(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        device="cpu"
    )

    test_prompts = [
        # Test 1: Flowchart simple
        "Create a Mermaid flowchart for a simple login process",
        
        # Test 2: Sequence diagram
        "Generate a Mermaid sequence diagram for an HTTP request/response",
        
        # Test 3: Architecture
        "Create a Mermaid diagram showing a 3-tier web architecture",
        
        # Test 4: Class diagram (plus complexe)
        "Generate a Mermaid class diagram for a simple e-commerce system",
    ]

    print("\n" + "="*60)
    print("STARTING ZERO-SHOT EVALUATION")
    print("="*60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        print("-" * 60)
        
        try:
            print("Generating... (Streaming output)")
            result = generator.generate_diagram(prompt, stream_output=True)
            mermaid_code = result["mermaid"]
            
            # print("GENERATED MERMAID:") # Already streamed
            # print(mermaid_code)
            
            # Basic validation
            is_valid, error = generator.validate_mermaid(mermaid_code)
            print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
            if not is_valid:
                print(f"Error: {error}")
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
        
        print("=" * 60)

if __name__ == "__main__":
    try:
        print("Starting Zero-Shot Test Script...")
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception("An unexpected error occurred:")
        sys.exit(1)

