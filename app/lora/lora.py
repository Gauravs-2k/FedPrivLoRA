import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

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

DEFAULT_TRAIN_LIMIT = 3000

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.model.inference import parse_dtype
from app.utils.config import settings


def _parse_list(value: Optional[str]):
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _prepare_it_support(example: Dict[str, object]) -> Dict[str, str]:
    subject = (example.get("subject") or "").strip()
    body = (example.get("body") or "").strip()
    prompt = f"{subject}\n\n{body}".strip()
    answer = (example.get("answer") or "").strip()
    return {"prompt": prompt, "response": answer}


def _prepare_finance(example: Dict[str, object]) -> Dict[str, str]:
    company = (example.get("COMPANY_ID") or "").strip()
    query = (example.get("QUERY") or "").strip()
    context = (example.get("CONTEXT") or "").strip()
    parts = []
    if company:
        parts.append(f"Company: {company}")
    if query:
        parts.append(f"Question: {query}")
    if context:
        parts.append(f"Context:\n{context}")
    prompt = "\n\n".join(parts)
    answer = (example.get("ANSWER") or "").strip()
    return {"prompt": prompt, "response": answer}


def _prepare_hr(example: Dict[str, object]) -> Dict[str, str]:
    conversation = example.get("messages") or []
    segments = []
    response = ""
    for message in conversation:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "assistant":
            response = content
            break
        label = "System" if role == "system" else "User"
        segments.append(f"{label}: {content}")
    prompt = "\n".join(segments)
    return {"prompt": prompt, "response": response}


def _prepare_engineering(example: Dict[str, object]) -> Dict[str, str]:
    prompt = (example.get("input") or "").strip()
    response = (example.get("output") or "").strip()
    return {"prompt": prompt, "response": response}


DEPARTMENT_DATASETS: Dict[str, Dict[str, object]] = {
    "it_support": {
        "dataset": "Tobi-Bueck/customer-support-tickets",
        "preprocess": _prepare_it_support,
        "text_field": "prompt",
        "response_field": "response",
        "train_split": "train",
        "max_train_samples": 10000,
    },
    "finance": {
        "dataset": "sweatSmile/FinanceQA",
        "preprocess": _prepare_finance,
        "text_field": "prompt",
        "response_field": "response",
        "train_split": "train",
        "max_train_samples": 10000,
    },
    "hr": {
        "dataset": "syncora/hr-policies-qa-dataset",
        "preprocess": _prepare_hr,
        "text_field": "prompt",
        "response_field": "response",
        "train_split": "train",
        "max_train_samples": 10000,
    },
    "engineering": {
        "dataset": "nvidia/OpenCodeInstruct",
        "preprocess": _prepare_engineering,
        "text_field": "prompt",
        "response_field": "response",
        "train_split": "train",
        "max_train_samples": 10000,
    },
}


def _tokenize_dataset(dataset, tokenizer, text_field, response_field, response_separator, max_length, num_proc):
    def formatter(example):
        if text_field not in example:
            raise ValueError(f"Field '{text_field}' not found in example")
        prompt = example[text_field]
        if response_field:
            if response_field not in example:
                raise ValueError(f"Field '{response_field}' not found in example")
            formatted = f"{prompt}{response_separator}{example[response_field]}"
        else:
            formatted = prompt
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
    parser.add_argument("--peft-output-dir", help="Override output directory")
    parser.add_argument("--departments", nargs="+", choices=sorted(DEPARTMENT_DATASETS.keys()))
    cli_args = parser.parse_args()

    # Create args object with hardcoded values
    class Args:
        def __init__(self):
            self.base_model = base_model
            self.response_separator = response_separator
            self.peft_output_dir = cli_args.peft_output_dir or peft_output_dir
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
            self.lora_target_modules = cli_args.lora_target_modules
            self.device_map = cli_args.device_map
            self.dtype = cli_args.dtype
            self.fp16 = fp16
            self.bf16 = False
            self.gradient_checkpointing = gradient_checkpointing
            self.seed = cli_args.seed
            self.max_train_samples = cli_args.max_train_samples
            self.max_eval_samples = None
            self.num_proc = cli_args.num_proc
            self.dataset_config = None

    args = Args()

    departments = cli_args.departments or list(DEPARTMENT_DATASETS.keys())

    for dept in departments:
        spec = DEPARTMENT_DATASETS[dept]
        dept_output_dir = f"{args.peft_output_dir}_{dept}"
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        train_single_department(dept, spec, dept_output_dir, args)


def train_single_department(department: str, spec: Dict[str, object], output_dir: str, args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    load_kwargs = {}
    if spec.get("dataset_config"):
        load_kwargs["name"] = spec["dataset_config"]
    if settings.HF_TOKEN:
        load_kwargs["token"] = settings.HF_TOKEN
    train_split = spec.get("train_split", "train")
    dataset = load_dataset(spec["dataset"], split=train_split, **load_kwargs)
    limit = DEFAULT_TRAIN_LIMIT
    spec_limit = spec.get("max_train_samples")
    if spec_limit:
        limit = min(limit, spec_limit)
    if args.max_train_samples:
        limit = min(limit, args.max_train_samples)
    if limit:
        dataset = dataset.select(range(min(len(dataset), limit)))

    preprocess: Optional[Callable[[Dict[str, object]], Dict[str, str]]] = spec.get("preprocess")
    if preprocess:
        map_kwargs = {
            "batched": False,
            "remove_columns": dataset.column_names,
        }
        if args.num_proc:
            map_kwargs["num_proc"] = args.num_proc
        dataset = dataset.map(preprocess, **map_kwargs)
    text_field = spec.get("text_field", "prompt")
    response_field = spec.get("response_field", "response")
    train_dataset = _tokenize_dataset(
        dataset,
        tokenizer,
        text_field,
        response_field,
        args.response_separator,
        args.max_length,
        args.num_proc,
    )

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
        eval_strategy="no",
        eval_steps=args.eval_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        save_total_limit=2,
        load_best_model_at_end=False,
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
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()


#  source env/bin/activate && PYTHONPATH=$PWD python app/lora/lora.py --departments engineering finance hr it_support