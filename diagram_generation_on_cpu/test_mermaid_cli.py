import logging
import sys
import os
import shutil

# Add current directory to path so we can import diagram_inference
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diagram_inference import DiagramGenerator

# Configure logging to show DEBUG messages (this will show if fallback is triggered)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logging.getLogger("diagram_inference").setLevel(logging.DEBUG)

class TestDiagramGenerator(DiagramGenerator):
    """Subclass to bypass model loading in __init__"""
    def __init__(self):
        pass

def main():
    print("=== Testing Mermaid CLI Integration ===")
    
    # DEBUG: Check environment
    print("\n[DEBUG] Environment Check:")
    print(f"OS Name: {os.name}")
    mmdc_which = shutil.which("mmdc")
    mmdc_cmd_which = shutil.which("mmdc.cmd")
    print(f"shutil.which('mmdc'): {mmdc_which}")
    print(f"shutil.which('mmdc.cmd'): {mmdc_cmd_which}")
    
    if os.name == 'nt':
        appdata = os.getenv('APPDATA')
        if appdata:
            manual_path = os.path.join(appdata, 'npm', 'mmdc.cmd')
            print(f"Manual check at {manual_path}: {os.path.exists(manual_path)}")
    
    # Initialize generator (using subclass to skip model load)
    generator = TestDiagramGenerator()
    
    # 1. Test Valid Diagram
    valid_mermaid = """graph TD
    A[Start] --> B{Is it working?}
    B -->|Yes| C[Great!]
    B -->|No| D[Debug]"""
    
    print("\n[Test 1] Validating Correct Diagram:")
    is_valid, msg = generator.validate_mermaid(valid_mermaid)
    print(f"Result: {'VALID' if is_valid else 'INVALID'}")
    if msg:
        print(f"Message: {msg}")
    
    # 2. Test Invalid Diagram (Syntax Error)
    # We need a syntax error that Basic Validation passes but Mermaid CLI catches.
    # Basic validation checks if the first line contains a valid type (e.g. "graph").
    # It does NOT check the direction (TD, LR, etc).
    # So "graph INVALID_DIRECTION" passes basic validation, but fails Mermaid CLI.
    
    invalid_mermaid = """graph INVALID_DIRECTION
    A --> B""" 
    # Basic validation (in diagram_inference.py) only checks:
    # 1. Valid type (graph TD is valid)
    # 2. Bracket counts (A[Start] and B[End] are balanced)
    # So Basic Validation should say TRUE (Valid).
    # Mermaid CLI should say FALSE (Error).
    
    print("\n[Test 2] Validating Diagram with Syntax Error (Invalid Arrow):")
    print(f"Code:\n{invalid_mermaid}")
    is_valid, msg = generator.validate_mermaid(invalid_mermaid)
    print(f"Result: {'VALID' if is_valid else 'INVALID'}")
    print(f"Message: {msg}")
    
    if not is_valid and "Mermaid CLI Error" in msg:
        print("\n[SUCCESS] The error message confirms Mermaid CLI was used!")
    elif is_valid:
        print("\n[FAILURE] The diagram was marked VALID. This means Basic Validation was likely used (it's too simple to catch this error), or Mermaid CLI failed to run.")
    else:
        print(f"\n[INFO] Validation failed with message: {msg}")

if __name__ == "__main__":
    main()