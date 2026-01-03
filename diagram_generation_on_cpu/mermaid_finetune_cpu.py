# ============================================================================
# Fine-Tuning Qwen2.5-1.5B for Mermaid Diagram Generation on CPU
# ============================================================================
# Purpose: Train a compact LLM to generate Mermaid syntax for architecture,
#          flowchart, sequence, and class diagrams from natural language
# Hardware: CPU-only (Intel NUC or similar, 32GB+ RAM recommended)
# Training Time: 1-3 hours for ~500 examples on CPU
# ============================================================================

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

# Setup logging immediately
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*60)
print("Initializing Fine-Tuning Script...")
print("Loading heavy libraries (Torch, Transformers, PEFT)... this may take a moment on CPU.")
print("="*60)

import torch
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

print("Libraries loaded successfully.")



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
        default="../gemini_fine_tune/dataset/training_data.jsonl",
        metadata={"help": "Path to training data (JSONL format)"}
    )
    eval_data_path: Optional[str] = field(
        default="../gemini_fine_tune/dataset/validation_data.jsonl",
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
    Supports both simple format and Gemini API format.
    """
    if not Path(data_path).exists():
        logger.warning(f"Data file {data_path} not found. Using synthetic examples.")
        return create_synthetic_dataset()
    
    data = []
    error_count = 0
    MAX_ERRORS = 5  # Stop after N errors to prevent training on bad data

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                    # Handle Gemini format
                    if "contents" in item:
                        instruction = item["contents"][0]["parts"][0]["text"]
                        output = item["contents"][1]["parts"][0]["text"]
                        data.append({"instruction": instruction, "output": output})
                    # Handle simple format
                    elif "instruction" in item and "output" in item:
                        data.append(item)
                    else:
                        raise ValueError("Missing required fields (instruction/output or contents)")
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error parsing line {line_num}: {e}")
                    if error_count > MAX_ERRORS:
                        raise ValueError(f"Too many errors ({error_count}) encountered while loading dataset. Aborting.")
                    continue
    
    if len(data) == 0:
        raise ValueError(f"No valid training examples found in {data_path}")

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
        dtype=torch.float32,  # float32 on CPU (float16/bfloat16 less helpful)
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
    
    # Update model config to match tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    
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
        outputs = tokenizer(
            examples["text"],
            padding="max_length",
            max_length=data_args.max_seq_length,
            truncation=True,
            return_tensors=None,
        )
        # For Causal LM, labels are usually the same as input_ids
        # We must replace padding token id's of the labels by -100 so it's ignored by the loss
        outputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in outputs["input_ids"]
        ]
        return outputs
    
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
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        # CRITICAL FIX: Ensure inputs require gradients when using checkpointing
        model.enable_input_require_grads()
        
    model.print_trainable_parameters()
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Training Configuration
    # ─────────────────────────────────────────────────────────────────────
    logger.info("Setting up trainer...")
    
    # Override critical settings for CPU
    # training_args.num_train_epochs = 3 # Use argument value
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
    training_args.use_cpu = True # Explicitly tell Trainer to use CPU
    
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
    
    # Disable cache to save memory during training
    model.config.use_cache = False
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Train!
    # ─────────────────────────────────────────────────────────────────────
    
    # Calculate training stats for user feedback
    num_update_steps_per_epoch = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    total_steps = int(num_update_steps_per_epoch * training_args.num_train_epochs)
    
    # Heuristic for CPU: ~60s per step (varies by hardware)
    estimated_seconds = total_steps * 60 
    estimated_hours = estimated_seconds / 3600
    
    print("\n" + "="*60)
    print("TRAINING ESTIMATION")
    print("="*60)
    print(f"Examples: {len(train_dataset)}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Batch Size (Effective): {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Total Optimization Steps: {total_steps}")
    print(f"Estimated Time (CPU): ~{estimated_hours:.1f} hours (@ 60s/step)")
    
    if estimated_hours > 4:
        print("\n⚠️  WARNING: Training will take a long time on CPU.")
        print("   Consider reducing dataset size or epochs for testing.")
        print("   You can use --max_steps 10 to run a quick test.")
    print("="*60 + "\n")

    logger.info("Starting training...")
    trainer.train()
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Save Model
    # ─────────────────────────────────────────────────────────────────────
    logger.info(f"Saving model to {training_args.output_dir}")
    
    # Re-enable cache for inference
    model.config.use_cache = True
    
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
