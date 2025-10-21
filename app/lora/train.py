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
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--peft-output-dir", default="lora-output")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--response-field")
    parser.add_argument("--prompt-template", default="{text}")
    parser.add_argument("--response-separator", default="\n")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--evaluation-strategy", default="no")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", choices=["none", "all", "lora_only"], default="none")
    parser.add_argument("--lora-target-modules")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--num-proc", type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    load_kwargs = {}
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config
    dataset = load_dataset(args.dataset, **load_kwargs)
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
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer))
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
        output_dir=args.peft_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        save_total_limit=2,
        load_best_model_at_end=args.evaluation_strategy != "no",
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
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
    trainer.save_model(args.peft_output_dir)
    tokenizer.save_pretrained(args.peft_output_dir)


if __name__ == "__main__":
    main()
