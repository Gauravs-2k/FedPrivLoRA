import argparse
import sys
from pathlib import Path
from typing import Optional
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.model.inference import parse_dtype


def _parse_list(value: Optional[str]):
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _tokenize_dataset(dataset, tokenizer, text_field, template, response_field, response_separator, max_length, num_proc):
    def formatter(example):
        if text_field not in example:
            raise ValueError(f"Field '{text_field}' not found in example")
        formatted = template.format(**example)
        if response_field:
            if response_field not in example:
                raise ValueError(f"Field '{response_field}' not found in example")
            formatted += response_separator + example[response_field]
        tokenized = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        return tokenized

    mapped = dataset.map(
        formatter,
        batched=False,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )
    return mapped


def main():
    # Hardcoded training parameters
    base_model = "Qwen/Qwen1.5-1.8B-Chat"
    dataset_path = "app/dataset/dept"
    text_field = "text"
    response_field = "response"
    response_separator = "\n### Response:\n"
    peft_output_dir = "qwen_dept_lora"
    num_train_epochs = 1
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 8
    learning_rate = 2e-4
    lora_r = 8
    lora_alpha = 16
    max_length = 256
    gradient_checkpointing = True
    fp16 = True
    
    # Optional parameters (can be modified if needed)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--lora-target-modules")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--num-proc", type=int)
    # Allow overriding dataset and output dir for testing individual departments
    parser.add_argument("--dataset", help="Override dataset path")
    parser.add_argument("--peft-output-dir", help="Override output directory")
    args = parser.parse_args()

    # Create args object with hardcoded values
    class Args:
        def __init__(self):
            self.base_model = base_model
            self.dataset = args.dataset or dataset_path  # Allow override
            self.text_field = text_field
            self.response_field = response_field
            self.prompt_template = "{text}"
            self.response_separator = response_separator
            self.peft_output_dir = args.peft_output_dir or peft_output_dir  # Allow override
            self.train_split = "train"
            self.eval_split = None
            self.max_length = max_length
            self.per_device_train_batch_size = per_device_train_batch_size
            self.per_device_eval_batch_size = 1
            self.gradient_accumulation_steps = gradient_accumulation_steps
            self.learning_rate = learning_rate
            self.weight_decay = 0.0
            self.num_train_epochs = num_train_epochs
            self.warmup_steps = 100
            self.logging_steps = 10
            self.save_steps = 500
            self.eval_steps = 500
            self.evaluation_strategy = "no"
            self.lora_r = lora_r
            self.lora_alpha = lora_alpha
            self.lora_dropout = 0.05
            self.lora_bias = "none"
            self.lora_target_modules = args.lora_target_modules
            self.device_map = args.device_map
            self.dtype = args.dtype
            self.fp16 = fp16
            self.bf16 = False
            self.gradient_checkpointing = gradient_checkpointing
            self.seed = args.seed
            self.max_train_samples = args.max_train_samples
            self.max_eval_samples = None
            self.num_proc = args.num_proc
            self.dataset_config = None

    args = Args()

    # Check if dataset is a directory with department files
    dataset_path = Path(args.dataset)
    if dataset_path.is_dir():
        # Find all department JSONL files
        dept_files = list(dataset_path.glob("*_dept.jsonl"))
        if not dept_files:
            raise ValueError(f"No department files found in {dataset_path}")
        
        print(f"Found {len(dept_files)} department files:")
        for f in dept_files:
            print(f"  - {f.name}")
        
        # Train each department separately
        for dept_file in dept_files:
            dept_name = dept_file.stem.replace("_dept", "").lower()
            dept_output_dir = f"{args.peft_output_dir}_{dept_name}"
            
            print(f"\n{'='*50}")
            print(f"Training LoRA for {dept_name.upper()} department")
            print(f"Dataset: {dept_file}")
            print(f"Output: {dept_output_dir}")
            print(f"{'='*50}")
            
            # Clear any cached models and GPU memory before training
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Train this department
            train_single_department(
                dept_file, 
                dept_output_dir, 
                args
            )
        
        print(f"\n{'='*50}")
        print("All department LoRA models trained successfully!")
        print(f"Models saved in {args.peft_output_dir}_<department>")
        print(f"{'='*50}")
        return
    
    # Single dataset training (original behavior)
    train_single_department(args.dataset, args.peft_output_dir, args)


def train_single_department(dataset_path, output_dir, args):
    """Train LoRA for a single dataset"""
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    load_kwargs = {}
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config
    dataset = load_dataset("json", data_files=str(dataset_path), **load_kwargs)
    if args.train_split not in dataset:
        raise ValueError(f"Split '{args.train_split}' not found in dataset")
    train_dataset = dataset[args.train_split]
    eval_dataset = None
    if args.eval_split:
        if args.eval_split not in dataset:
            raise ValueError(f"Split '{args.eval_split}' not found in dataset")
        eval_dataset = dataset[args.eval_split]

    train_dataset = _tokenize_dataset(
        train_dataset,
        tokenizer,
        args.text_field,
        args.prompt_template,
        args.response_field,
        args.response_separator,
        args.max_length,
        args.num_proc,
    )
    if eval_dataset is not None:
        eval_dataset = _tokenize_dataset(
            eval_dataset,
            tokenizer,
            args.text_field,
            args.prompt_template,
            args.response_field,
            args.response_separator,
            args.max_length,
            args.num_proc,
        )

    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.max_train_samples)))
    if eval_dataset is not None and args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.max_eval_samples)))

    torch_dtype = parse_dtype(args.dtype)
    model_kwargs = {"trust_remote_code": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    # Don't use device_map for now to avoid accelerate issues
    # if args.device_map.lower() != "none":
    #     model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_targets = _parse_list(args.lora_target_modules)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        save_total_limit=2,
        load_best_model_at_end=args.evaluation_strategy != "no",
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        # Explicitly set device to avoid accelerate issues
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
