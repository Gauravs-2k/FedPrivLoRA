import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_dataset
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


def build_tokenizer(model_name: str, token: str | None):
	kwargs = {"trust_remote_code": True}
	if token:
		kwargs["token"] = token
	tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
	if tokenizer.pad_token is None:
		tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
	return tokenizer


def build_model(model_name: str, dtype: str, device_map: str, token: str | None, vocab_size: int):
	kwargs = {"trust_remote_code": True}
	parsed_dtype = parse_dtype(dtype)
	if parsed_dtype is not None:
		kwargs["torch_dtype"] = parsed_dtype
	if device_map.lower() != "none":
		kwargs["device_map"] = device_map
	if token:
		kwargs["token"] = token
	model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
	model.resize_token_embeddings(vocab_size)
	return model


def capture_initial_state(model: AutoModelForCausalLM):
	state = {}
	for name, param in model.named_parameters():
		state[name] = param.detach().cpu().clone()
	for name, buffer in model.named_buffers():
		state[name] = buffer.detach().cpu().clone()
	return state


def compute_residual(initial_state, model: AutoModelForCausalLM):
	residual = {}
	for name, param in model.named_parameters():
		if name in initial_state:
			current = param.detach().cpu()
			residual[name] = current - initial_state[name]
	for name, buffer in model.named_buffers():
		if name in initial_state:
			current = buffer.detach().cpu()
			residual[name] = current - initial_state[name]
	return residual


def prepare_dataset(tokenizer, args, token: str | None):
	kwargs = {}
	if token:
		kwargs["token"] = token
	dataset = load_dataset(args.dataset, split=args.split, **kwargs)
	if args.max_train_samples and args.max_train_samples < len(dataset):
		dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
	def format_example(example):
		text = example["chosen"]
		if args.append_eos and tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
			text = text + tokenizer.eos_token
		return {"text": text}
	dataset = dataset.map(format_example, remove_columns=dataset.column_names)
	def tokenize_batch(batch):
		encodings = tokenizer(batch["text"], truncation=True, padding=True, max_length=args.max_length)
		encodings["labels"] = encodings["input_ids"].copy()
		return encodings
	map_kwargs = {"batched": True, "remove_columns": ["text"]}
	if args.num_proc:
		map_kwargs["num_proc"] = args.num_proc
	return dataset.map(tokenize_batch, **map_kwargs)


def run_training():
	token = settings.HF_TOKEN
	base_model_name = "Qwen/Qwen2.5-0.5B"
	dataset_name = "Anthropic/hh-rlhf"
	max_samples = 1000
	max_length = 512
	seed_value = 42
	tokenizer = build_tokenizer(base_model_name, token)
	model = build_model(base_model_name, "float32", "auto", token, len(tokenizer))
	model.gradient_checkpointing_enable()
	model.config.use_cache = False
	initial_state = capture_initial_state(model)
	args = SimpleNamespace(
		dataset=dataset_name,
		split="train",
		max_train_samples=max_samples,
		max_length=max_length,
		num_proc=None,
		seed=seed_value,
		append_eos=False,
	)
	train_dataset = prepare_dataset(tokenizer, args, token)
	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
	training_args = TrainingArguments(
		output_dir="residual_model",
		num_train_epochs=1.0,
		per_device_train_batch_size=1,
		gradient_accumulation_steps=16,
		learning_rate=5e-6,
		weight_decay=0.0,
		warmup_steps=100,
		logging_steps=10,
		save_steps=500,
		report_to="none",
		seed=seed_value,
		fp16=False,
		bf16=False,
		gradient_checkpointing=True,
		max_steps=-1,
		max_grad_norm=0.5,
		optim="adamw_bnb_8bit",
	)
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		data_collator=data_collator,
		tokenizer=tokenizer,
	)
	trainer.train()
	trainer.save_model("residual_model")
	residual = compute_residual(initial_state, model)
	torch.save(residual, "instruction_residuals.pt")
	stats = {
		"samples": len(train_dataset),
		"output": "instruction_residuals.pt",
	}
	Path("instruction_residuals_meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
	print(json.dumps(stats, indent=2))


def main():
	run_training()


if __name__ == "__main__":
	main()

