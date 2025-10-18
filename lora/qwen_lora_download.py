import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


def parse_dtype(value: str):
    if value.lower() == "auto":
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


def resolve_modules(model, provided):
    if provided:
        return provided
    suffixes = {name.split(".")[-1] for name, _ in model.named_modules() if name}
    candidates = [
        ["c_attn"],
        ["W_pack"],
        ["query_key_value"],
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    ]
    for option in candidates:
        if set(option).issubset(suffixes):
            return option
    raise ValueError("Unable to resolve target modules")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen1.5-1.8B")
    parser.add_argument("--output-dir", default="qwen_lora")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--target-modules", nargs="+")
    parser.add_argument("--token")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = parse_dtype(args.dtype)
    kwargs = {"trust_remote_code": True}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    device_map_value = args.device_map.lower()
    if device_map_value not in {"auto", "none"}:
        kwargs["device_map"] = args.device_map
    if args.token:
        kwargs["token"] = args.token
    tokenizer_kwargs = {"trust_remote_code": True}
    if args.token:
        tokenizer_kwargs["token"] = args.token
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    modules = resolve_modules(model, args.target_modules)
    config = LoraConfig(r=16, lora_alpha=32, target_modules=modules, task_type=TaskType.CAUSAL_LM)
    lora_model = get_peft_model(model, config)
    lora_model = lora_model.to("cpu")
    tokenizer.save_pretrained(output_dir)
    lora_model.save_pretrained(output_dir)
    print(f"Saved tokenizer and LoRA model to {output_dir}")


if __name__ == "__main__":
    main()
