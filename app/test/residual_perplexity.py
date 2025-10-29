import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
	sys.path.append(str(ROOT_DIR))

from app.lora.instruction_residual_lora import DEPARTMENT_DATASETS
from app.model.inference import parse_dtype
from app.utils.config import settings
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model(name: str, dtype: str, device_map: str, token: str | None):
	kwargs = {"trust_remote_code": True}
	parsed = parse_dtype(dtype)
	if parsed is not None:
		kwargs["torch_dtype"] = parsed
	if device_map.lower() != "none":
		kwargs["device_map"] = device_map
	if token:
		kwargs["token"] = token
	model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
	model.eval()
	return model


def load_tokenizer(name: str, token: str | None):
	kwargs = {"trust_remote_code": True}
	if token:
		kwargs["token"] = token
	return AutoTokenizer.from_pretrained(name, **kwargs)


def apply_residual(model, residual_state):
	params = dict(model.named_parameters())
	buffers = dict(model.named_buffers())
	missing = []
	mismatched = []
	with torch.no_grad():
		for name, delta in residual_state.items():
			target = params.get(name)
			if target is None:
				target = buffers.get(name)
			if target is None:
				missing.append(name)
				continue
			source = target.data if isinstance(target, torch.nn.Parameter) else target
			if source.shape != delta.shape:
				mismatched.append(name)
				continue
			source.add_(delta.to(device=source.device, dtype=source.dtype))
	return missing, mismatched


def load_residual_state(path: str | None, repo_id: str | None, filename: str, token: str | None):
	if path:
		resolved = Path(path)
		if resolved.is_file():
			return torch.load(resolved, map_location="cpu")
		if resolved.is_dir():
			candidate = resolved / filename
			if candidate.is_file():
				return torch.load(candidate, map_location="cpu")
	if not repo_id:
		raise FileNotFoundError("Residual source not provided")
	download_kwargs = {"repo_id": repo_id, "filename": filename, "repo_type": "model"}
	if token:
		download_kwargs["token"] = token
	artifact = hf_hub_download(**download_kwargs)
	return torch.load(Path(artifact), map_location="cpu")


def load_dataset_samples(spec, token: str | None):
	dataset_kwargs = {}
	if token:
		dataset_kwargs["token"] = token
	if spec.dataset_config:
		dataset_kwargs["name"] = spec.dataset_config
	try:
		dataset = load_dataset(spec.dataset, split=spec.split, **dataset_kwargs)
	except Exception as error:
		if "gated" in str(error).lower() and not token:
			raise RuntimeError(f"Dataset {spec.dataset} requires an authenticated token") from error
		raise
	limit = spec.max_samples
	preprocess = getattr(spec, "preprocess", None)
	samples = []
	for record in dataset:
		processed = preprocess(record) if preprocess else record
		if not processed:
			continue
		prompt = processed.get(spec.text_field, "")
		if not prompt:
			continue
		response = processed.get(spec.response_field, "") if spec.response_field else ""
		combined = f"{prompt}\n{response}".strip() if response else prompt
		if combined:
			samples.append(combined)
		if limit and len(samples) >= limit:
			break
	return samples


def calculate_perplexity(model, tokenizer, text: str, max_length: int, stride: int):
	device = model.device if hasattr(model, "device") else next(model.parameters()).device
	encodings = tokenizer(text, return_tensors="pt")
	max_len = encodings.input_ids.size(1)
	nlls = []
	prev_end = 0
	for start in range(0, max_len, stride):
		end = min(start + max_length, max_len)
		target_len = end - prev_end
		input_ids = encodings.input_ids[:, start:end].to(device)
		target_ids = input_ids.clone()
		if target_len < input_ids.size(1):
			target_ids[:, :-target_len] = -100
		with torch.no_grad():
			outputs = model(input_ids, labels=target_ids)
		nlls.append(outputs.loss.detach())
		prev_end = end
		if end == max_len:
			break
	if not nlls:
		return float("nan")
	return torch.exp(torch.stack(nlls).mean()).item()


def evaluate_model(model, tokenizer, samples, max_length: int, stride: int):
	if not samples:
		return float("nan")
	values = []
	for sample in samples:
		values.append(calculate_perplexity(model, tokenizer, sample, max_length, stride))
	return float(torch.tensor(values).mean().item())


def run_evaluation(base_config, dataset_spec, residual_state):
	token = settings.HF_TOKEN
	samples = load_dataset_samples(dataset_spec, token)
	if not samples:
		raise RuntimeError("Dataset yielded no samples")
	base_tokenizer = load_tokenizer(base_config.base_model, token)
	base_model = load_base_model(base_config.base_model, base_config.dtype, base_config.device_map, token)
	base_perplexity = evaluate_model(base_model, base_tokenizer, samples, base_config.max_length, base_config.stride)
	missing, mismatched = apply_residual(base_model, residual_state)
	residual_perplexity = evaluate_model(base_model, base_tokenizer, samples, base_config.max_length, base_config.stride)
	return {
		"label": dataset_spec.label,
		"dataset": dataset_spec.dataset,
		"dataset_config": dataset_spec.dataset_config,
		"split": dataset_spec.split,
		"sample_count": len(samples),
		"base_perplexity": base_perplexity,
		"residual_perplexity": residual_perplexity,
		"delta": residual_perplexity - base_perplexity,
		"missing_keys": missing,
		"mismatched_keys": mismatched,
	}


def build_dataset_specs():
	training_spec = SimpleNamespace(
		label="training",
		dataset="Anthropic/hh-rlhf",
		dataset_config=None,
		split="train",
		text_field="chosen",
		response_field=None,
		preprocess=None,
		max_samples=1000,
	)
	specs = [training_spec]
	for label, definition in DEPARTMENT_DATASETS.items():
		specs.append(
			SimpleNamespace(
				label=label,
				dataset=definition["dataset"],
				dataset_config=definition.get("dataset_config"),
				split=definition.get("train_split", "train"),
				text_field=definition.get("text_field", "prompt"),
				response_field=definition.get("response_field"),
				preprocess=definition.get("preprocess"),
				max_samples=min(definition.get("max_train_samples", 256) or 256, 256),
			)
		)
	return specs


def main():
	base_config = SimpleNamespace(
		residual=str(Path(__file__).resolve().parents[2] / "instruction_residuals.pt"),
		residual_repo="Gaurav2k/qwen2-0.5b-instruction-residuals",
		residual_filename="qwen2-0.5b-instruction-residuals.pt",
		base_model="Qwen/Qwen2-0.5B",
		dtype="float32",
		device_map="auto",
		max_length=512,
		stride=512,
		output=str(Path(__file__).resolve().parents[2] / "results" / "residual_perplexity_results.json"),
	)
	token = settings.HF_TOKEN
	residual_state = load_residual_state(base_config.residual, base_config.residual_repo, base_config.residual_filename, token)
	results = []
	for spec in build_dataset_specs():
		result = run_evaluation(base_config, spec, residual_state)
		results.append(result)
	output_path = Path(base_config.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
	print(json.dumps(results, indent=2))
	print(f"Saved results to {output_path}")


if __name__ == "__main__":
	main()
