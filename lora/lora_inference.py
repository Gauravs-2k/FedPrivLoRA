import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen1.5-1.8B")
    parser.add_argument("--peft-dir", default="qwen_lora")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()
    
    hf_token = os.environ.get("HF_TOKEN")
    dtype = parse_dtype(args.dtype)
    base_kwargs = {"trust_remote_code": True}
    if dtype is not None:
        base_kwargs["torch_dtype"] = dtype
    if args.device_map.lower() != "none":
        base_kwargs["device_map"] = args.device_map
    if hf_token:
        base_kwargs["token"] = hf_token
    peft_config_path = os.path.join(args.peft_dir, "adapter_config.json")
    use_peft = bool(args.peft_dir) and os.path.isdir(args.peft_dir) and os.path.exists(peft_config_path)
    tokenizer_source = args.peft_dir if use_peft else args.base_model
    tokenizer_kwargs = {"trust_remote_code": True}
    if hf_token and tokenizer_source == args.base_model:
        tokenizer_kwargs["token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **base_kwargs)
    model = base_model
    if use_peft:
        try:
            model = PeftModel.from_pretrained(base_model, args.peft_dir)
        except (OSError, ValueError):
            print(f"Unable to load PEFT adapters from {args.peft_dir}, falling back to base model.")
            model = base_model
    model.eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(text)


if __name__ == "__main__":
    main()
