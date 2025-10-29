import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	Trainer,
	TrainingArguments,
)
from transformers import DataCollatorForSeq2Seq

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


def _parse_bool(value: Optional[str], default: bool):
	if value is None:
		return default
	normalized = value.strip().lower()
	if normalized in {"1", "true", "yes", "y"}:
		return True
	if normalized in {"0", "false", "no", "n"}:
		return False
	raise ValueError(f"Cannot parse boolean value from '{value}'")


def _prepare_it_support(example):
	subject = (example.get("subject") or "").strip()
	body = (example.get("body") or "").strip()
	if subject and body:
		prompt = f"{subject}\n\n{body}"
	else:
		prompt = subject or body
	answer = (example.get("answer") or "").strip()
	return {"prompt": prompt, "response": answer}


def _prepare_finance(example):
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


def _prepare_hr(example):
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


def _prepare_engineering(example):
	prompt = (example.get("input") or "").strip()
	response = (example.get("output") or "").strip()
	return {"prompt": prompt, "response": response}


DEPARTMENT_DATASETS = {
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


def _tokenize_dataset(dataset, tokenizer, text_field, template, response_field, response_separator, max_length, num_proc):
    def formatter(example):
        if text_field not in example or response_field not in example:
            raise ValueError(f"Missing fields: {text_field}, {response_field}")
        
        prompt = example[text_field]
        response = example[response_field]
        
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
        input_ids = prompt_tokens + [tokenizer.eos_token_id] + response_tokens + [tokenizer.eos_token_id]
        
        # Truncate
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        labels = ([-100] * (len(prompt_tokens) + 1) + 
                 response_tokens + [tokenizer.eos_token_id])
        
        if len(labels) > max_length:
            labels = labels[:max_length]
        
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    return dataset.map(
        formatter,
        batched=False,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )



def load_instruction_residual(path: Optional[str], repo_id: Optional[str], filename: str, token: Optional[str]):
	if path:
		resolved = Path(path)
		if resolved.is_file():
			return torch.load(resolved, map_location="cpu")
		if resolved.is_dir():
			local_candidate = resolved / filename
			if local_candidate.is_file():
				return torch.load(local_candidate, map_location="cpu")
	if not repo_id:
		return None
	download_kwargs = {"repo_id": repo_id, "filename": filename, "repo_type": "model"}
	if token:
		download_kwargs["token"] = token
	downloaded = hf_hub_download(**download_kwargs)
	return torch.load(Path(downloaded), map_location="cpu")


def apply_instruction_residual(model, residual_state):
	if residual_state is None:
		return
	params = dict(model.named_parameters())
	buffers = dict(model.named_buffers())
	with torch.no_grad():
		for name, delta in residual_state.items():
			if name in params:
				target = params[name].data
				target.add_(delta.to(device=target.device, dtype=target.dtype))
			elif name in buffers:
				target = buffers[name]
				target.add_(delta.to(device=target.device, dtype=target.dtype))


def train_single_department(dataset_path, output_dir, args, residual_state):
	tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id

	load_kwargs = {}
	if args.dataset_config:
		load_kwargs["name"] = args.dataset_config
	if args.hf_token:
		load_kwargs["token"] = args.hf_token
	train_dataset = load_dataset(dataset_path, split=args.train_split, **load_kwargs)
	eval_dataset = None
	if args.eval_split:
		eval_dataset = load_dataset(dataset_path, split=args.eval_split, **load_kwargs)
	preprocess_fn = getattr(args, "preprocess", None)
	if preprocess_fn:
		train_columns = train_dataset.column_names
		map_kwargs = {"batched": False, "remove_columns": train_columns}
		if args.num_proc:
			map_kwargs["num_proc"] = args.num_proc
		train_dataset = train_dataset.map(preprocess_fn, **map_kwargs)
		if eval_dataset is not None:
			eval_columns = eval_dataset.column_names
			eval_kwargs = {"batched": False, "remove_columns": eval_columns}
			if args.num_proc:
				eval_kwargs["num_proc"] = args.num_proc
			eval_dataset = eval_dataset.map(preprocess_fn, **eval_kwargs)

	train_dataset = _tokenize_dataset(
		train_dataset,
		tokenizer,
		args.text_field,
		None,
		args.response_field,
		None,
		args.max_length,
		args.num_proc,
	)
	if eval_dataset is not None:
		eval_dataset = _tokenize_dataset(
			eval_dataset,
			tokenizer,
			args.text_field,
			None,
			args.response_field,
			None,
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
	model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
	apply_instruction_residual(model, residual_state)

	if torch.cuda.is_available():
		model = model.to("cuda")

	model.config.use_cache = False
	if args.gradient_checkpointing:
		model.gradient_checkpointing_enable()

	lora_targets = _parse_list(args.lora_target_modules)
	if not lora_targets:
		lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
	lora_config = LoraConfig(
		r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		bias=args.lora_bias,
		task_type="CAUSAL_LM",
		target_modules=lora_targets,
	)
	model = get_peft_model(model, lora_config)
	model.print_trainable_parameters()

	data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
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



def main():
	base_model = "Qwen/Qwen2-0.5B"
	dataset_path = "it_support"
	text_field = "text"
	response_field = "response"
	response_separator = "\n### Response:\n"
	peft_output_dir = "qwen_dept_lora_instruction"
	num_train_epochs = 3
	per_device_train_batch_size = 2
	gradient_accumulation_steps = 8
	learning_rate = 5e-5
	lora_r = 16
	lora_alpha = 32
	max_length = 512
	gradient_checkpointing = True
	fp16 = True
	bf16 = False
	warmup_steps = 50
	weight_decay = 0.01
	instruction_residual_path = None
	instruction_residual_repo = "Gaurav2k/qwen2-0.5b-instruction-residuals"
	instruction_residual_filename = "qwen2-0.5b-instruction-residuals.pt"

	class Args:
		def __init__(self):
			self.base_model = base_model
			self.dataset = dataset_path
			self.text_field = text_field
			self.response_field = response_field
			self.prompt_template = "{text}"
			self.response_separator = response_separator
			self.peft_output_dir = peft_output_dir
			self.train_split = "train"
			self.eval_split = None
			self.max_length = max_length
			self.per_device_train_batch_size = per_device_train_batch_size
			self.per_device_eval_batch_size = 1
			self.gradient_accumulation_steps = gradient_accumulation_steps
			self.learning_rate = learning_rate
			self.weight_decay = weight_decay
			self.num_train_epochs = num_train_epochs
			self.warmup_steps = warmup_steps
			self.logging_steps = 10
			self.save_steps = 500
			self.eval_steps = 500
			self.evaluation_strategy = "no"
			self.lora_r = lora_r
			self.lora_alpha = lora_alpha
			self.lora_dropout = 0.05
			self.lora_bias = "none"
			self.lora_target_modules = None
			self.device_map = "auto"
			self.dtype = "auto"
			self.fp16 = fp16
			self.bf16 = bf16
			self.gradient_checkpointing = gradient_checkpointing
			self.seed = 42
			self.max_train_samples = None
			self.max_eval_samples = None
			self.num_proc = None
			self.dataset_config = None
			self.instruction_residual_path = str(instruction_residual_path) if instruction_residual_path else None
			self.instruction_residual_repo = instruction_residual_repo
			self.instruction_residual_filename = instruction_residual_filename
			self.hf_token = settings.HF_TOKEN
			self.preprocess = None

	config = Args()
	residual_state = load_instruction_residual(
		config.instruction_residual_path,
		config.instruction_residual_repo,
		config.instruction_residual_filename,
		config.hf_token,
	)

	selected_dataset = config.dataset.lower()
	if selected_dataset in DEPARTMENT_DATASETS:
		spec = DEPARTMENT_DATASETS[selected_dataset]
		dept_args = deepcopy(config)
		dept_args.dataset = spec["dataset"]
		dept_args.text_field = spec.get("text_field", dept_args.text_field)
		dept_args.response_field = spec.get("response_field", dept_args.response_field)
		dept_args.prompt_template = spec.get("prompt_template", dept_args.prompt_template)
		dept_args.response_separator = spec.get("response_separator", dept_args.response_separator)
		dept_args.train_split = spec.get("train_split", dept_args.train_split)
		dept_args.eval_split = spec.get("eval_split", dept_args.eval_split)
		dept_args.dataset_config = spec.get("dataset_config", dept_args.dataset_config)
		dept_args.preprocess = spec.get("preprocess")
		if "max_length" in spec:
			dept_args.max_length = spec["max_length"]
		if "gradient_accumulation_steps" in spec:
			dept_args.gradient_accumulation_steps = spec["gradient_accumulation_steps"]
		if "max_train_samples" in spec:
			dept_args.max_train_samples = spec["max_train_samples"]
		if "max_eval_samples" in spec:
			dept_args.max_eval_samples = spec["max_eval_samples"]
		dept_output_dir = spec.get("output_dir", f"{config.peft_output_dir}_{selected_dataset}")
		dept_args.peft_output_dir = dept_output_dir
		train_single_department(dept_args.dataset, dept_output_dir, dept_args, residual_state)
		return

	train_single_department(config.dataset, config.peft_output_dir, config, residual_state)


if __name__ == "__main__":
	main()
