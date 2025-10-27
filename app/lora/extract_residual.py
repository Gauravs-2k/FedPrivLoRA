import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
	sys.path.append(str(ROOT_DIR))

from app.model.inference import parse_dtype
from app.utils.config import settings


def load_model(name: str, dtype: str, device_map: str, token: str | None):
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


def capture_state(model: AutoModelForCausalLM):
	state = {}
	for name, param in model.named_parameters():
		state[name] = param.detach().cpu().clone()
	for name, buffer in model.named_buffers():
		state[name] = buffer.detach().cpu().clone()
	return state


def compute_residual(base_state: dict[str, torch.Tensor], reference: AutoModelForCausalLM):
	residual = {}
	missing = []
	mismatched = []
	for name, param in reference.named_parameters():
		baseline = base_state.get(name)
		if baseline is None:
			missing.append(name)
			continue
		current = param.detach().cpu()
		if current.shape != baseline.shape:
			mismatched.append(name)
			continue
		residual[name] = current - baseline
	for name, buffer in reference.named_buffers():
		baseline = base_state.get(name)
		if baseline is None:
			missing.append(name)
			continue
		current = buffer.detach().cpu()
		if current.shape != baseline.shape:
			mismatched.append(name)
			continue
		residual[name] = current - baseline
	return residual, missing, mismatched


def run_extraction():
	config = SimpleNamespace(
		base_model="Qwen/Qwen2-0.5B",
		instruct_model="Qwen/Qwen2-0.5B-Instruct",
		dtype="float32",
		device_map="cpu",
		output=ROOT_DIR / "instruction_residuals.pt",
	)
	token = settings.HF_TOKEN
	print("Loading tokenizers...")
	base_tokenizer = load_tokenizer(config.base_model, token)
	instruct_tokenizer = load_tokenizer(config.instruct_model, token)
	base_vocab = len(base_tokenizer)
	instruct_vocab = len(instruct_tokenizer)
	if base_vocab != instruct_vocab:
		print(f"Warning: vocab mismatch base={base_vocab} instruct={instruct_vocab}")
	print(f"Loading {config.base_model}...")
	base_model = load_model(config.base_model, config.dtype, config.device_map, token)
	base_state = capture_state(base_model)
	print(f"Loading {config.instruct_model}...")
	instruct_model = load_model(config.instruct_model, config.dtype, config.device_map, token)
	print("Extracting residuals...")
	residual, missing, mismatched = compute_residual(base_state, instruct_model)
	embed_key = "model.embed_tokens.weight"
	embed_shapes_match = (
		base_model.get_input_embeddings().weight.shape == instruct_model.get_input_embeddings().weight.shape
	)
	embed_vocab_match = (
		base_model.get_input_embeddings().weight.shape[0] == base_vocab
		and instruct_model.get_input_embeddings().weight.shape[0] == instruct_vocab
	)
	embedding_skipped = False
	if embed_key in residual and (not embed_shapes_match or not embed_vocab_match):
		print("Skipping embedding layer due to size mismatch")
		residual.pop(embed_key, None)
		mismatched = [name for name in mismatched if name != embed_key]
		embedding_skipped = True
	torch.save(residual, config.output)
	stats = {
		"base_model": config.base_model,
		"instruct_model": config.instruct_model,
		"parameters": len(residual),
		"missing_count": len(missing),
		"mismatched_count": len(mismatched),
		"base_vocab": base_vocab,
		"instruct_vocab": instruct_vocab,
		"embedding_skipped": embedding_skipped,
		"output": str(config.output),
	}
	if missing:
		stats["missing"] = missing
	if mismatched:
		stats["mismatched"] = mismatched
	meta_path = ROOT_DIR / "instruction_residuals_meta.json"
	meta_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
	print(json.dumps(stats, indent=2))
	print(f"Saved residuals to {config.output}")
	if missing:
		print(f"Missing keys: {len(missing)}")
	if mismatched:
		print(f"Mismatched keys: {len(mismatched)}")


def main():
	run_extraction()


if __name__ == "__main__":
	main()
