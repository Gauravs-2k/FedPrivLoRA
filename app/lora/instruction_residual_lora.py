import argparse
import gc
import sys
from copy import deepcopy
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
		"prompt_template": "{prompt}",
		"train_split": "train",
		"max_train_samples": 1000,
	},
	"finance": {
		"dataset": "sweatSmile/FinanceQA",
		"preprocess": _prepare_finance,
		"text_field": "prompt",
		"response_field": "response",
		"prompt_template": "{prompt}",
		"train_split": "train",
		"max_train_samples": 1000,
	},
	"hr": {
		"dataset": "syncora/hr-policies-qa-dataset",
		"preprocess": _prepare_hr,
		"text_field": "prompt",
		"response_field": "response",
		"prompt_template": "{prompt}",
		"train_split": "train",
		"max_train_samples": 1000,
	},
	"engineering": {
		"dataset": "nvidia/OpenCodeInstruct",
		"preprocess": _prepare_engineering,
		"text_field": "prompt",
		"response_field": "response",
		"prompt_template": "{prompt}",
		"train_split": "train",
		"max_train_samples": 1000,
	},
}


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


def load_instruction_residual(path: Optional[str]):
	if not path:
		return None
	resolved = Path(path)
	if not resolved.is_file():
		raise FileNotFoundError(f"Instruction residuals not found at {resolved}")
	return torch.load(resolved, map_location="cpu")


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
		tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

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
	model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
	model.resize_token_embeddings(len(tokenizer))
	apply_instruction_residual(model, residual_state)

	if torch.cuda.is_available():
		model = model.to("cuda")

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
		evaluation_strategy=args.evaluation_strategy,
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
	base_model = "Qwen/Qwen1.5-1.8B"
	dataset_path = "it_support"
	text_field = "text"
	response_field = "response"
	response_separator = "\n### Response:\n"
	peft_output_dir = "qwen_dept_lora_instruction"
	num_train_epochs = 1
	per_device_train_batch_size = 1
	gradient_accumulation_steps = 8
	learning_rate = 2e-4
	lora_r = 8
	lora_alpha = 16
	max_length = 256
	gradient_checkpointing = True
	fp16 = True
	bf16 = False
	instruction_residual_path = ROOT_DIR / "instruction_residuals.pt"

	parser = argparse.ArgumentParser()
	parser.add_argument("--device-map", default="auto")
	parser.add_argument("--dtype", default="auto")
	parser.add_argument("--lora-target-modules")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--max-train-samples", type=int)
	parser.add_argument("--max-eval-samples", type=int)
	parser.add_argument("--num-proc", type=int)
	parser.add_argument("--dataset")
	parser.add_argument("--peft-output-dir")
	parser.add_argument("--base-model")
	parser.add_argument("--instruction-residual-path")
	parser.add_argument("--num-train-epochs", type=float)
	parser.add_argument("--per-device-train-batch-size", type=int)
	parser.add_argument("--per-device-eval-batch-size", type=int)
	parser.add_argument("--gradient-accumulation-steps", type=int)
	parser.add_argument("--learning-rate", type=float)
	parser.add_argument("--lora-r", type=int)
	parser.add_argument("--lora-alpha", type=int)
	parser.add_argument("--lora-dropout", type=float)
	parser.add_argument("--max-length", type=int)
	parser.add_argument("--logging-steps", type=int)
	parser.add_argument("--warmup-steps", type=int)
	parser.add_argument("--save-steps", type=int)
	parser.add_argument("--eval-steps", type=int)
	parser.add_argument("--evaluation-strategy")
	parser.add_argument("--fp16")
	parser.add_argument("--bf16")
	parser.add_argument("--gradient-checkpointing")
	parser.add_argument("--train-split")
	parser.add_argument("--eval-split")
	parser.add_argument("--dataset-config")
	args = parser.parse_args()

	class Args:
		def __init__(self):
			self.base_model = args.base_model or base_model
			self.dataset = args.dataset or dataset_path
			self.text_field = text_field
			self.response_field = response_field
			self.prompt_template = "{text}"
			self.response_separator = response_separator
			self.peft_output_dir = args.peft_output_dir or peft_output_dir
			self.train_split = args.train_split or "train"
			self.eval_split = args.eval_split
			self.max_length = args.max_length or max_length
			self.per_device_train_batch_size = args.per_device_train_batch_size or per_device_train_batch_size
			self.per_device_eval_batch_size = args.per_device_eval_batch_size or 1
			self.gradient_accumulation_steps = args.gradient_accumulation_steps or gradient_accumulation_steps
			self.learning_rate = args.learning_rate or learning_rate
			self.weight_decay = 0.0
			self.num_train_epochs = args.num_train_epochs or num_train_epochs
			self.warmup_steps = args.warmup_steps or 100
			self.logging_steps = args.logging_steps or 10
			self.save_steps = args.save_steps or 500
			self.eval_steps = args.eval_steps or 500
			self.evaluation_strategy = args.evaluation_strategy or "no"
			self.lora_r = args.lora_r or lora_r
			self.lora_alpha = args.lora_alpha or lora_alpha
			self.lora_dropout = args.lora_dropout or 0.05
			self.lora_bias = "none"
			self.lora_target_modules = args.lora_target_modules
			self.device_map = args.device_map
			self.dtype = args.dtype
			self.fp16 = _parse_bool(args.fp16, fp16)
			self.bf16 = _parse_bool(args.bf16, bf16)
			self.gradient_checkpointing = _parse_bool(args.gradient_checkpointing, gradient_checkpointing)
			self.seed = args.seed
			self.max_train_samples = args.max_train_samples
			self.max_eval_samples = args.max_eval_samples
			self.num_proc = args.num_proc
			self.dataset_config = args.dataset_config
			self.instruction_residual_path = args.instruction_residual_path or str(instruction_residual_path)
			self.hf_token = settings.HF_TOKEN
			self.preprocess = None

	config = Args()
	residual_state = load_instruction_residual(config.instruction_residual_path)

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
