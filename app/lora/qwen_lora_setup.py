import argparse
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


def parse_dtype(value: str) -> Optional[torch.dtype]:
    if not value or value.lower() == "auto":
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


def resolve_target_modules(model: AutoModelForCausalLM, override: Optional[List[str]]) -> List[str]:
    if override:
        return override
    suffixes = {name.split(".")[-1] for name, _ in model.named_modules() if name}
    options = [
        ["c_attn"],
        ["W_pack"],
        ["query_key_value"],
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    ]
    for option in options:
        if set(option).issubset(suffixes):
            return option
    raise ValueError("Unable to infer target modules. Provide --target-modules explicitly.")


def load_model(
    model_name: str,
    torch_dtype: Optional[torch.dtype],
    device_map: Optional[str],
    target_modules: Optional[List[str]],
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if device_map:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    modules = resolve_target_modules(model, target_modules)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=modules,
        task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(model, lora_config)
    return tokenizer, lora_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen1.5-1.8B")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--target-modules", nargs="+")
    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)
    device_map = args.device_map
    if device_map and device_map.lower() == "none":
        device_map = None
    tokenizer, model = load_model(args.model_name, dtype, device_map, args.target_modules)
    model.print_trainable_parameters()
    config = model.peft_config["default"]
    print(f"Target modules: {config.target_modules}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
