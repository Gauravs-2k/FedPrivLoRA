import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from perplexity_plot import plot_heatmap, plot_accuracy


def build_departments(root: Path) -> Dict[str, Dict[str, Path]]:
	return {
		"finance": {
			"lora": root / "qwen_dept_lora_finance",
			"dataset": root / "app" / "dataset" / "dept" / "FINANCE_dept.jsonl",
		},
		"hr": {
			"lora": root / "qwen_dept_lora_hr",
			"dataset": root / "app" / "dataset" / "dept" / "HR_dept.jsonl",
		},
		"it_support": {
			"lora": root / "qwen_dept_lora_it_support",
			"dataset": root / "app" / "dataset" / "dept" / "IT_SUPPORT_dept.jsonl",
		},
		"engineering": {
			"lora": root / "qwen_dept_lora_engineering",
			"dataset": root / "app" / "dataset" / "dept" / "ENGINEERING_dept.jsonl",
		},
	}


def load_samples(path: Path, limit: int | None) -> List[str]:
	samples: List[str] = []
	if not path.exists():
		return samples
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			data = json.loads(line)
			question = str(data.get("text", "")).strip()
			answer = str(data.get("response", "")).strip()
			if question or answer:
				samples.append(f"{question}\n{answer}".strip())
			if limit and len(samples) >= limit:
				break
	return samples


def calculate_perplexity(model, tokenizer, text: str, max_length: int, stride: int) -> float:
	device = model.device if hasattr(model, "device") else next(model.parameters()).device
	encodings = tokenizer(text, return_tensors="pt")
	max_len = encodings.input_ids.size(1)
	nlls: List[torch.Tensor] = []
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


def evaluate_model(model, tokenizer, samples: List[str], max_length: int, stride: int) -> float:
	if not samples:
		return float("nan")
	values: List[float] = []
	for sample in samples:
		values.append(calculate_perplexity(model, tokenizer, sample, max_length, stride))
	return float(torch.tensor(values).mean().item())


def parse_dtype(value: str):
	if value is None or value.lower() == "auto":
		return None
	mapping = {
		"float16": torch.float16,
		"fp16": torch.float16,
		"bfloat16": torch.bfloat16,
		"bf16": torch.bfloat16,
		"float32": torch.float32,
		"fp32": torch.float32,
	}
	key = value.lower()
	if key not in mapping:
		raise ValueError(f"Unsupported dtype {value}")
	return mapping[key]


def load_base_model(name: str, dtype: str, device_map: str):
	kwargs = {"trust_remote_code": True}
	parsed = parse_dtype(dtype)
	if parsed is not None:
		kwargs["torch_dtype"] = parsed
	if device_map != "auto":
		kwargs["device_map"] = device_map
	model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
	model.eval()
	return model


def load_tokenizer(source: str | Path):
	return AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)


def load_lora_model(base_name: str, lora_path: Path, dtype: str, device_map: str, vocab_size: int | None):
	model = load_base_model(base_name, dtype, device_map)
	if vocab_size is not None:
		embeddings = model.get_input_embeddings()
		if embeddings is not None and embeddings.weight.size(0) != vocab_size:
			model.resize_token_embeddings(vocab_size)
	lora_model = PeftModel.from_pretrained(model, str(lora_path))
	lora_model.eval()
	return lora_model


def benchmark(base_model_name: str, departments: Dict[str, Dict[str, Path]], dtype: str, device_map: str, limit: int, max_length: int, stride: int):
	base_tokenizer = load_tokenizer(base_model_name)
	base_model = load_base_model(base_model_name, dtype, device_map)
	samples = {name: load_samples(meta["dataset"], limit) for name, meta in departments.items()}
	base_scores = {}
	for name, texts in samples.items():
		print(f"Evaluating base model on {name} ({len(texts)} samples)")
		base_scores[name] = evaluate_model(base_model, base_tokenizer, texts, max_length, stride)
	results = {"base_model": base_scores, "lora_models": {}, "accuracy": {}}
	del base_model
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	for name, meta in departments.items():
		lora_path = meta["lora"]
		if not lora_path.exists():
			print(f"Skipping {name} LoRA: missing directory")
			continue
		print(f"Evaluating {name} LoRA")
		try:
			lora_tokenizer = load_tokenizer(lora_path)
		except OSError:
			lora_tokenizer = base_tokenizer
		vocab_size = len(lora_tokenizer)
		model = load_lora_model(base_model_name, lora_path, dtype, device_map, vocab_size)
		scores = {}
		wins = 0
		total = 0
		for target_name, texts in samples.items():
			score = evaluate_model(model, lora_tokenizer, texts, max_length, stride)
			scores[target_name] = score
			base_value = results["base_model"].get(target_name)
			if base_value is not None and not math.isnan(base_value):
				total += 1
				if score < base_value:
					wins += 1
		accuracy = (wins / total) if total else float("nan")
		results["lora_models"][name] = scores
		results["accuracy"][name] = accuracy
		del model
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	return results


def save_json(data, path: Path):
	path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--base-model", default="Qwen/Qwen1.5-1.8B-Chat")
	parser.add_argument("--dtype", default="float16")
	parser.add_argument("--device-map", default="auto")
	parser.add_argument("--max-samples", type=int, default=50)
	parser.add_argument("--max-length", type=int, default=512)
	parser.add_argument("--stride", type=int, default=512)
	parser.add_argument("--output-json", default="perplexity_benchmark_results.json")
	parser.add_argument("--heatmap", default="perplexity_heatmap.png")
	parser.add_argument("--accuracy-plot", default="perplexity_accuracy.png")
	parser.add_argument("--load-from-json", action="store_true", help="Load results from existing JSON file instead of running benchmark")
	args = parser.parse_args()
	root = Path(__file__).resolve().parents[2]
	departments = build_departments(root)
	if args.load_from_json:
		output_json = Path(args.output_json)
		if not output_json.exists():
			print(f"JSON file {output_json} does not exist")
			return
		results = json.loads(output_json.read_text())
	else:
		results = benchmark(
			args.base_model,
			departments,
			args.dtype,
			args.device_map,
			args.max_samples if args.max_samples > 0 else None,
			args.max_length,
			args.stride,
		)
		output_json = Path(args.output_json)
		save_json(results, output_json)
		print(f"Saved results to {output_json}")
	heatmap_path = Path(args.heatmap)
	plot_heatmap(results, list(departments.keys()), heatmap_path)
	if heatmap_path.exists():
		print(f"Saved heatmap to {heatmap_path}")
	accuracy_path = Path(args.accuracy_plot)
	plot_accuracy(results, accuracy_path)
	if accuracy_path.exists():
		print(f"Saved accuracy plot to {accuracy_path}")


if __name__ == "__main__":
	main()
