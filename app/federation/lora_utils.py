from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.model.inference import parse_dtype

DEFAULT_BASE_MODEL = "Qwen/Qwen1.5-1.8B-Chat"
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES: Optional[Sequence[str]] = None
DEFAULT_DTYPE = "auto"
DEFAULT_DEVICE_MAP = "cpu"


def create_tokenizer(base_model: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    return tokenizer


def create_lora_model(
    base_model: str = DEFAULT_BASE_MODEL,
    r: int = DEFAULT_LORA_R,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    *,
    target_modules: Optional[Sequence[str]] = DEFAULT_LORA_TARGET_MODULES,
    dtype: str = DEFAULT_DTYPE,
    device_map: str = DEFAULT_DEVICE_MAP,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = create_tokenizer(base_model)
    kwargs: Dict[str, object] = {"trust_remote_code": True}
    torch_dtype = parse_dtype(dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device_map.lower() != "none":
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model.resize_token_embeddings(len(tokenizer))
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules) if target_modules is not None else None,
    )
    peft_model = get_peft_model(model, config)
    return peft_model, tokenizer


def collect_lora_parameter_names(model: torch.nn.Module) -> List[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def collect_lora_state(model: torch.nn.Module, names: Sequence[str]) -> OrderedDict[str, torch.Tensor]:
    state = OrderedDict()
    full_state = model.state_dict()
    for name in names:
        state[name] = full_state[name].detach().clone()
    return state


def apply_lora_state(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    target_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            target = target_params[name]
            target.copy_(tensor.to(target.device))


def lora_state_to_ndarrays(state: Dict[str, torch.Tensor], names: Sequence[str]) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []
    for name in names:
        tensor = state[name].detach().cpu().numpy().astype(np.float32, copy=True)
        arrays.append(tensor)
    return arrays


def ndarrays_to_lora_state(names: Sequence[str], arrays: Sequence[np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, array in zip(names, arrays):
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)
        state[name] = tensor
    return state


def clone_numpy_state(arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [np.array(array, copy=True) for array in arrays]


def average_ndarrays(items: Sequence[Tuple[Sequence[np.ndarray], int]]) -> List[np.ndarray]:
    total_weight = sum(weight for _, weight in items)
    if total_weight == 0:
        return [np.array(arr, copy=True) for arr in items[0][0]]
    accumulators = [np.zeros_like(arr, dtype=np.float32) for arr in items[0][0]]
    for arrays, weight in items:
        fraction = weight / total_weight
        for index, array in enumerate(arrays):
            accumulators[index] += array.astype(np.float32, copy=False) * fraction
    return accumulators


def export_lora_adapter(
    base_model: str,
    names: Sequence[str],
    arrays: Sequence[np.ndarray],
    output_dir: Path,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Optional[Sequence[str]] = DEFAULT_LORA_TARGET_MODULES,
    dtype: str = DEFAULT_DTYPE,
    device_map: str = DEFAULT_DEVICE_MAP,
) -> None:
    model, tokenizer = create_lora_model(
        base_model,
        r,
        alpha,
        dropout,
        target_modules=target_modules,
        dtype=dtype,
        device_map=device_map,
    )
    device = next(model.parameters()).device
    state = ndarrays_to_lora_state(names, arrays, device)
    apply_lora_state(model, state)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
