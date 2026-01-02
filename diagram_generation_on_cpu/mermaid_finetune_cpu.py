# ============================================================================
# Fine-Tuning Qwen2.5-1.5B for Mermaid Diagram Generation on CPU
# ============================================================================
# Purpose: Train a compact LLM to generate Mermaid syntax for architecture,
#          flowchart, sequence, and class diagrams from natural language
# Hardware: CPU-only (Intel NUC or similar, 32GB+ RAM recommended)
# Training Time: 1-3 hours for ~500 examples on CPU
# ============================================================================

import os
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. CONFIGURATION & ARGUMENTS
# ============================================================================

@dataclass
class ModelArguments:
    """Model-specific arguments."""
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for model downloads"}
    )
    

@dataclass
class DataArguments:
    """Data-specific arguments."""
    train_data_path: str = field(
        default="train_data.json",
        metadata={"help": "Path to training data (JSONL format)"}
    )
    eval_data_path: Optional[str] = field(
        default="eval_data.json",
        metadata={"help": "Path to evaluation data (JSONL format)"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length (keep small for CPU)"}
    )
    

@dataclass
class LoraArguments:
    """LoRA-specific arguments."""
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank (4-16 typical, lower for CPU)"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA scaling factor (2x rank is common)"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={"help": "Target modules for LoRA adaptation"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Whether to include bias in LoRA layers"}
    )


# ============================================================================
# 2. DATASET LOADING & FORMATTING
# ============================================================================

def load_training_data(data_path: str) -> Dataset:
    """
    Load training data from JSONL file.
    
    Expected format per line:
    {
        "instruction": "Create a flowchart for a login process",
        "output": "graph TD\n    A[Start] --> B[Enter Username]\n    B --> C[Enter Password]\n    ...",
        "category": "flowchart"  # Optional: for filtering
    }
    """
    if not Path(data_path).exists():
        logger.warning(f"Data file {data_path} not found. Using synthetic examples.")
        return create_synthetic_dataset()
    
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    logger.info(f"Loaded {len(data)} examples from {data_path}")
    return Dataset.from_list(data)


def create_synthetic_dataset() -> Dataset:
    """
    Create small synthetic dataset for testing.
    Replace with real data from diagram_llm2 or Mermaid datasets.
    """
    examples = [
        {
            "instruction": "Create a flowchart for a simple login process",
            "output": "graph TD\n    A[Start] --> B[Enter Username]\n    B --> C[Enter Password]\n    C --> D{Valid?}\n    D -->|Yes| E[Login Success]\n    D -->|No| F[Show Error]\n    F --> B"
        },
        {
            "instruction": "Design a sequence diagram for user registration",
            "output": "sequenceDiagram\n    participant User\n    participant App\n    participant DB\n    User->>App: Click Register\n    App->>User: Show Form\n    User->>App: Submit Details\n    App->>DB: Save User\n    DB-->>App: Confirm\n    App-->>User: Success"
        },
        {
            "instruction": "Create an architecture diagram showing a 3-tier web application",
            "output": "graph TB\n    Client[Client Browser]\n    LB[Load Balancer]\n    Web1[Web Server 1]\n    Web2[Web Server 2]\n    API[API Gateway]\n    DB[(Database)]\n    Cache[Cache Layer]\n    Client --> LB\n    LB --> Web1\n    LB --> Web2\n    Web1 --> API\n    Web2 --> API\n    API --> Cache\n    API --> DB\n    Cache --> DB"
        },
    ]
    return Dataset.from_list(examples)


def format_examples(examples: Dict) -> Dict:
    """
    Format raw examples into instruction-response pairs with strict templates.
    
    This is KEY: ensuring model learns to ALWAYS output properly formatted Mermaid.
    """
    formatted_texts = []
    
    for instruction, output in zip(examples['instruction'], examples['output']):
        # Define a strict template the model should follow
        prompt = f"""Below is a request to generate a diagram in Mermaid syntax.

### Request:
{instruction}

### Mermaid Diagram:
"""
        
        # Full text = prompt + output (for causal LM training)
        full_text = prompt + output
        formatted_texts.append(full_text)
    
    return {"text": formatted_texts}


# ============================================================================
# 3. MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training pipeline."""
    
    # Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    
    # You can also pass command-line args:
    # python script.py --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    #                  --train_data_path train.jsonl \
    #                  --output_dir ./lora_mermaid \
    #                  --num_train_epochs 3 \
    #                  --per_device_train_batch_size 1
    
    model_args, data_args, training_args, lora_args = \
        parser.parse_args_into_dataclasses()
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Load Model & Tokenizer
    # ─────────────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
    # For CPU-only, explicitly set device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="cpu",  # CPU-only training
        torch_dtype=torch.float32,  # float32 on CPU (float16/bfloat16 less helpful)
        cache_dir=model_args.cache_dir,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="right",
        use_fast=True,
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Load & Format Data
    # ─────────────────────────────────────────────────────────────────────
    logger.info(f"Loading training data from {data_args.train_data_path}")
    train_dataset = load_training_data(data_args.train_data_path)
    
    # Format for training
    train_dataset = train_dataset.map(
        format_examples,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    # Load eval dataset if provided
    eval_dataset = None
    if data_args.eval_data_path and Path(data_args.eval_data_path).exists():
        logger.info(f"Loading eval data from {data_args.eval_data_path}")
        eval_dataset = load_training_data(data_args.eval_data_path)
        eval_dataset = eval_dataset.map(
            format_examples,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            max_length=data_args.max_seq_length,
            truncation=True,
            return_tensors=None,
        )
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
    
    logger.info(f"Tokenized dataset. Example: {train_dataset[0]}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Configure LoRA
    # ─────────────────────────────────────────────────────────────────────
    logger.info("Configuring LoRA...")
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Training Configuration
    # ─────────────────────────────────────────────────────────────────────
    logger.info("Setting up trainer...")
    
    # Override critical settings for CPU
    training_args.num_train_epochs = 3
    training_args.per_device_train_batch_size = 1  # CPU constraint
    training_args.gradient_accumulation_steps = 4  # Simulate batch size 4
    training_args.learning_rate = 2e-4
    training_args.warmup_steps = 10
    training_args.save_steps = 50
    training_args.eval_steps = 50
    training_args.logging_steps = 10
    training_args.max_steps = -1  # Use num_train_epochs instead
    training_args.fp16 = False  # No FP16 on CPU
    training_args.bf16 = False
    training_args.optim = "adamw_torch"  # CPU-compatible optimizer
    
    # Data collator for causal LM
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Train!
    # ─────────────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train()
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Save Model
    # ─────────────────────────────────────────────────────────────────────
    logger.info(f"Saving model to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
