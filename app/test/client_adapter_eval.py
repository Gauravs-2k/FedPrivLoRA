import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import Dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.federation.department_client import _format_training_example
from app.federation.lora_utils import DEFAULT_BASE_MODEL, DEFAULT_DEVICE_MAP, DEFAULT_DTYPE
from app.model.inference import parse_dtype


def load_records(path: Path, limit: int | None) -> List[str]:
	records: List[str] = []
	if not path.exists():
		return records
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			if limit is not None and len(records) >= limit:
				break
			line = line.strip()
			if not line:
				continue
			try:
				payload = json.loads(line)
			except json.JSONDecodeError:
				continue
			user = str(payload.get("text") or "").strip()
			assistant = str(payload.get("response") or "").strip()
			if not user or not assistant:
				continue
			records.append(_format_training_example({"user": user, "assistant": assistant}))
	return records


def build_dataset(texts: List[str], tokenizer: AutoTokenizer, max_seq_length: int) -> Dataset:
	dataset = Dataset.from_dict({"text": texts})

	def tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
		encodings = tokenizer(
			batch["text"],
			truncation=True,
			max_length=max_seq_length,
			padding="max_length",
		)
		return encodings

	return dataset.map(tokenize, batched=True, remove_columns=["text"])


def load_base_model(base_model: str, dtype: str, device_map: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
	kwargs: Dict[str, object] = {"trust_remote_code": True}
	parsed = parse_dtype(dtype)
	if parsed is not None:
		kwargs["torch_dtype"] = parsed
	if device_map.lower() != "none":
		kwargs["device_map"] = device_map
	model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
	tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
	model.resize_token_embeddings(len(tokenizer))
	model.eval()
	return model, tokenizer


def load_adapter_model(base_model: str, adapter_path: Path, dtype: str, device_map: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
	kwargs: Dict[str, object] = {"trust_remote_code": True}
	parsed = parse_dtype(dtype)
	if parsed is not None:
		kwargs["torch_dtype"] = parsed
	if device_map.lower() != "none":
		kwargs["device_map"] = device_map
	base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
	try:
		tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
	except OSError:
		tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
	base.resize_token_embeddings(len(tokenizer))
	model = PeftModel.from_pretrained(base, str(adapter_path))
	model.eval()
	return model, tokenizer


def evaluate_model(records: List[str], model: torch.nn.Module, tokenizer: AutoTokenizer, batch_size: int, max_seq_length: int) -> Tuple[float, float]:
	if not records:
		return float("nan"), float("nan")
	dataset = build_dataset(records, tokenizer, max_seq_length)
	dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
	loader = DataLoader(dataset, batch_size=batch_size)
	device = next(model.parameters()).device
	total_loss = 0.0
	total_tokens = 0
	model.eval()
	with torch.no_grad():
		for batch in loader:
			batch = {key: value.to(device) for key, value in batch.items()}
			labels = batch["input_ids"].clone()
			outputs = model(**batch, labels=labels)
			loss = outputs.loss
			tokens = labels.numel()
			total_loss += loss.item() * tokens
			total_tokens += tokens
	avg_loss = total_loss / total_tokens if total_tokens else float("nan")
	perplexity = math.exp(avg_loss) if total_tokens else float("nan")
	return avg_loss, perplexity


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
	parser.add_argument("--adapters-root", type=Path, default=Path("results") / "client_exports")
	parser.add_argument("--dataset-root", type=Path, default=Path("app") / "dataset" / "personal_clients")
	parser.add_argument("--dtype", default=DEFAULT_DTYPE)
	parser.add_argument("--device-map", default=DEFAULT_DEVICE_MAP)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--max-seq-length", type=int, default=256)
	parser.add_argument("--limit", type=int, default=0)
	parser.add_argument("--output", type=Path)
	parser.add_argument("--compare-base", action="store_true")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	limit = args.limit if args.limit > 0 else None
	if not args.adapters_root.exists():
		print(f"Adapters root {args.adapters_root} does not exist")
		return 1
	results: Dict[str, Dict[str, object]] = {}
	records_map: Dict[str, List[str]] = {}
	for adapter_path in sorted(args.adapters_root.iterdir()):
		if not adapter_path.is_dir():
			continue
		name = adapter_path.name
		dataset_path = args.dataset_root / f"{name}.jsonl"
		records = load_records(dataset_path, limit)
		if not records:
			print(f"Skipping {name}: no records")
			continue
		records_map[name] = records
		results[name] = {"samples": len(records)}
	if not records_map:
		if args.output:
			args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
		return 0
	if args.compare_base:
		base_model, base_tokenizer = load_base_model(args.base_model, args.dtype, args.device_map)
		for name, records in records_map.items():
			loss, perplexity = evaluate_model(records, base_model, base_tokenizer, args.batch_size, args.max_seq_length)
			print(f"[base] {name}: loss={loss:.4f} perplexity={perplexity:.4f} samples={len(records)}")
			results[name]["base"] = {"loss": loss, "perplexity": perplexity}
		del base_model
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	for adapter_path in sorted(args.adapters_root.iterdir()):
		if not adapter_path.is_dir():
			continue
		name = adapter_path.name
		records = records_map.get(name)
		if not records:
			continue
		model, tokenizer = load_adapter_model(args.base_model, adapter_path, args.dtype, args.device_map)
		loss, perplexity = evaluate_model(records, model, tokenizer, args.batch_size, args.max_seq_length)
		entry = results.setdefault(name, {"samples": len(records)})
		entry["adapter"] = {"loss": loss, "perplexity": perplexity}
		if "base" in entry:
			delta_loss = loss - float(entry["base"]["loss"])
			delta_ppl = perplexity - float(entry["base"]["perplexity"])
			entry["delta"] = {"loss": delta_loss, "perplexity": delta_ppl}
			delta_msg = f" delta_loss={delta_loss:+.4f} delta_ppl={delta_ppl:+.4f}"
		else:
			delta_msg = ""
		print(f"[adapter] {name}: loss={loss:.4f} perplexity={perplexity:.4f} samples={len(records)}{delta_msg}")
		del model
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	if args.output:
		args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


# PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /home/espresso/work/DML/DML-project/env/bin/python app/test/client_adapter_eval.py --device-map auto --dtype float16 --compare-base --output results/client_adapter_metrics.json