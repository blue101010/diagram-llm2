#!/bin/bash
# Launcher for CPU Fine-Tuning (Qwen2.5-1.5B)
# Usage: ./run_finetune.sh [arguments]

echo "============================================================"
echo "Starting CPU Fine-Tuning (Qwen2.5-1.5B)"
echo "============================================================"

# Determine Python path (Windows Git Bash vs Linux/Mac)
if [ -f "venv/Scripts/python" ]; then
    PYTHON_EXEC="venv/Scripts/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_EXEC="venv/bin/python"
else
    echo "ERROR: Virtual environment not found."
    echo "Please run: python -m venv venv && pip install -r requirements.txt"
    exit 1
fi

echo "Using interpreter: $PYTHON_EXEC"
"$PYTHON_EXEC" mermaid_finetune_cpu.py "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "An error occurred during execution."
    exit 1
fi

echo ""
echo "Processing complete."
