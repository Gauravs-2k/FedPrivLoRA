import argparse
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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


SERVER_DEFAULTS = {
    "base_model": "Qwen/Qwen1.5-1.8B-Chat",
    "peft_dir": "app/lora/qwen_lora",
    "dtype": "auto",
    "device_map": "auto",
    "max_new_tokens": 128,
}


@dataclass
class ModelEntry:
    model: torch.nn.Module
    tokenizers: Dict[str, AutoTokenizer]
    loaded_adapters: set
    active_adapter: Optional[str]


_MODEL_ENTRIES: Dict[tuple, ModelEntry] = {}


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = None
    base_model: Optional[str] = None
    peft_dir: Optional[str] = None
    dtype: Optional[str] = None
    device_map: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    peft_dir: Optional[str] = None
    dtype: Optional[str] = None
    device_map: Optional[str] = None


@dataclass
class GenerationResult:
    text: str
    completion_text: str
    prompt_tokens: int
    completion_tokens: int


app = FastAPI()


def _load_model(base_model: str, peft_dir: str, dtype: str, device_map: str, tokenizer=None):
    hf_token = os.environ.get("HF_TOKEN")
    parsed_dtype = parse_dtype(dtype)
    base_kwargs = {"trust_remote_code": True}
    if parsed_dtype is not None:
        base_kwargs["torch_dtype"] = parsed_dtype
    if device_map.lower() != "none":
        base_kwargs["device_map"] = device_map
    if hf_token:
        base_kwargs["token"] = hf_token
    base_model_instance = AutoModelForCausalLM.from_pretrained(base_model, **base_kwargs)

    peft_config_path = os.path.join(peft_dir, "adapter_config.json")
    use_peft = bool(peft_dir) and os.path.isdir(peft_dir) and os.path.exists(peft_config_path)
    model = base_model_instance

    if use_peft:
        if tokenizer is None:
            tokenizer = _load_tokenizer(base_model, peft_dir)
        model.resize_token_embeddings(len(tokenizer))
        adapter_name = Path(peft_dir).name
        try:
            model = PeftModel.from_pretrained(base_model_instance, peft_dir, adapter_name=adapter_name, is_trainable=False)
        except (OSError, ValueError) as e:
            print(f"Warning: Failed to load PeftModel: {e}. Using base model only.")
            model = base_model_instance

    model.eval()
    return model


def _load_tokenizer(base_model: str, peft_dir: str):
    hf_token = os.environ.get("HF_TOKEN")
    peft_config_path = os.path.join(peft_dir, "adapter_config.json")
    use_peft = bool(peft_dir) and os.path.isdir(peft_dir) and os.path.exists(peft_config_path)
    tokenizer_source = peft_dir if use_peft else base_model
    tokenizer_kwargs = {"trust_remote_code": True}
    if hf_token and tokenizer_source == base_model:
        tokenizer_kwargs["token"] = hf_token
    return AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)


def _cache_key(base_model: str, dtype: str, device_map: str):
    return (base_model, dtype or "auto", device_map or "auto")


def _get_model_and_tokenizer(base_model: str, peft_dir: str, dtype: str, device_map: str):
    key = _cache_key(base_model, dtype, device_map)
    adapter_name = Path(peft_dir).name if peft_dir else "__base__"
    entry = _MODEL_ENTRIES.get(key)
    if entry is None:
        tokenizer = _load_tokenizer(base_model, peft_dir)
        model = _load_model(base_model, peft_dir, dtype, device_map, tokenizer)
        loaded_adapters = set()
        active_adapter: Optional[str] = None
        if peft_dir and isinstance(model, PeftModel):
            loaded_adapters.add(adapter_name)
            active_adapter = adapter_name
        entry = ModelEntry(model=model.eval(), tokenizers={adapter_name: tokenizer}, loaded_adapters=loaded_adapters, active_adapter=active_adapter)
        _MODEL_ENTRIES[key] = entry
        return entry.model, tokenizer
    tokenizer = entry.tokenizers.get(adapter_name)
    if tokenizer is None:
        tokenizer = _load_tokenizer(base_model, peft_dir)
        entry.tokenizers[adapter_name] = tokenizer
    if peft_dir:
        if adapter_name not in entry.loaded_adapters:
            if isinstance(entry.model, PeftModel):
                entry.model.load_adapter(peft_dir, adapter_name=adapter_name, is_trainable=False)
            else:
                entry.model = PeftModel.from_pretrained(entry.model, peft_dir, adapter_name=adapter_name, is_trainable=False)
            entry.loaded_adapters.add(adapter_name)
        if isinstance(entry.model, PeftModel):
            entry.model.set_adapter(adapter_name)
        entry.active_adapter = adapter_name
    else:
        if isinstance(entry.model, PeftModel):
            disable_adapter = getattr(entry.model, "disable_adapter", None)
            if callable(disable_adapter):
                disable_adapter()
        entry.active_adapter = None
    entry.model.eval()
    if tokenizer is not None and hasattr(entry.model, "resize_token_embeddings"):
        entry.model.resize_token_embeddings(len(tokenizer))
    return entry.model, tokenizer


def reset_cache_for(peft_dir: Optional[str] = None) -> None:
    if not peft_dir:
        _MODEL_ENTRIES.clear()
        return
    adapter_name = Path(peft_dir).name
    for key in list(_MODEL_ENTRIES.keys()):
        entry = _MODEL_ENTRIES.get(key)
        if entry is None:
            continue
        if adapter_name in entry.loaded_adapters or adapter_name == "":
            _MODEL_ENTRIES.pop(key, None)


def _conversation_to_prompt(messages: List[ChatMessage], tokenizer: AutoTokenizer) -> str:
    structured = [{"role": msg.role, "content": msg.content} for msg in messages]
    try:
        return tokenizer.apply_chat_template(structured, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for msg in structured:
            parts.append(f"{msg['role'].upper()}: {msg['content']}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)


def generate_text(
    prompt: str,
    max_new_tokens: int,
    base_model: str,
    peft_dir: str,
    dtype: str,
    device_map: str,
    *,
    return_full_text: bool = True,
    **generation_kwargs,
) -> GenerationResult:
    model, tokenizer = _get_model_and_tokenizer(base_model, peft_dir, dtype, device_map)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_params: Dict[str, Optional[float]] = {"max_new_tokens": max_new_tokens}
    for key, value in generation_kwargs.items():
        if value is not None:
            generation_params[key] = value
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_params)
    prompt_token_count = inputs.input_ids.shape[-1]
    generated_ids = output_ids[:, prompt_token_count:]
    completion_tokens = generated_ids.shape[-1]
    completion_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] if completion_tokens > 0 else ""
    decode_ids = output_ids if return_full_text else generated_ids
    decoded_text = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)[0]
    return GenerationResult(
        text=decoded_text,
        completion_text=completion_text,
        prompt_tokens=prompt_token_count,
        completion_tokens=completion_tokens,
    )


@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    defaults = SERVER_DEFAULTS
    base_model = req.base_model or defaults["base_model"]
    peft_dir = req.peft_dir or defaults["peft_dir"]
    dtype = req.dtype or defaults["dtype"]
    device_map = req.device_map or defaults["device_map"]
    max_new_tokens = req.max_new_tokens or defaults["max_new_tokens"]
    try:
        result = generate_text(
            req.prompt,
            max_new_tokens,
            base_model,
            peft_dir,
            dtype,
            device_map,
            return_full_text=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    usage = {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }
    return {"text": result.text, "usage": usage}


@app.post("/v1/chat/completions")
def chat_completions_endpoint(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    defaults = SERVER_DEFAULTS
    base_model = req.model or defaults["base_model"]
    peft_dir = req.peft_dir or defaults["peft_dir"]
    dtype = req.dtype or defaults["dtype"]
    device_map = req.device_map or defaults["device_map"]
    max_new_tokens = req.max_tokens or defaults["max_new_tokens"]
    temperature = req.temperature if req.temperature is not None else 0.7
    top_p = req.top_p
    prompt_builder_model = _get_model_and_tokenizer(base_model, peft_dir, dtype, device_map)[1]
    prompt = _conversation_to_prompt(req.messages, prompt_builder_model)
    generation_args = {
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0 or (top_p is not None and top_p < 1.0),
    }
    try:
        result = generate_text(
            prompt,
            max_new_tokens,
            base_model,
            peft_dir,
            dtype,
            device_map,
            return_full_text=False,
            **generation_args,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    created_ts = int(time.time())
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created_ts,
        "model": base_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen1.5-1.8B-Chat")
    parser.add_argument("--peft-dir", default="qwen_lora")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server_defaults = {
        "base_model": args.base_model,
        "peft_dir": args.peft_dir,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_new_tokens": args.max_new_tokens,
    }
    SERVER_DEFAULTS.update(server_defaults)
    if args.serve:
        _get_model_and_tokenizer(args.base_model, args.peft_dir, args.dtype, args.device_map)
        uvicorn.run(app, host=args.host, port=args.port)
        return
    result = generate_text(
        args.prompt,
        args.max_new_tokens,
        args.base_model,
        args.peft_dir,
        args.dtype,
        args.device_map,
        return_full_text=True,
    )
    print(result.text)


if __name__ == "__main__":
    main()
