import os
import sys
from pathlib import Path
import argparse
import json
from typing import List, Optional
from huggingface_hub import snapshot_download

import streamlit as st
import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

KNOWN_DEPARTMENTS = ["engineering", "finance", "hr", "it_support"]

from app.utils.config import settings
from app.dataset.chat_logs import append_chat_record
from model.inference import (
    SERVER_DEFAULTS,
    ChatMessage,
    _conversation_to_prompt,
    _get_model_and_tokenizer,
    reset_cache_for,
    generate_text,
)


def _resolve_defaults():
    return {
        "base_model": SERVER_DEFAULTS["base_model"],
        "peft_dir": SERVER_DEFAULTS["peft_dir"],
        "dtype": SERVER_DEFAULTS["dtype"],
        "device_map": SERVER_DEFAULTS["device_map"],
        "max_new_tokens": SERVER_DEFAULTS["max_new_tokens"],
        "temperature": 0.7,
        "top_p": 1.0,
    }


def _adapter_root() -> Path:
    return CURRENT_DIR.parent / "results" / "adapters"


def _get_available_models():
    models = {}
    root = _adapter_root()
    if root.exists():
        for item in root.iterdir():
            if not item.is_dir():
                continue
            if item.name not in KNOWN_DEPARTMENTS:
                continue
            display = item.name.replace("_", " ").title()
            models[item.name] = (str(item), f"{display} Client", False, item.name)
    for department in KNOWN_DEPARTMENTS:
        if department in models:
            continue
        display = department.replace("_", " ").title()
        repo = settings.ADAPTER_REPOS.get(department)
        if repo:
            models[department] = (repo, f"{display} Client", True, department)
        else:
            models[department] = ("", f"{display} Client", False, department)
    return [models[name] for name in sorted(models.keys())]


def _ensure_adapter(identifier: str, is_remote: bool, department: Optional[str]):
    if not identifier or not department:
        return "", None
    if not is_remote:
        path = Path(identifier)
        if path.is_dir():
            return str(path), None
        return "", f"Adapter directory '{identifier}' not found."
    target_dir = _adapter_root() / department
    config_path = target_dir / "adapter_config.json"
    if target_dir.exists() and config_path.exists():
        return str(target_dir), None
    target_dir.mkdir(parents=True, exist_ok=True)
    download_kwargs = {
        "repo_id": identifier,
        "local_dir": str(target_dir),
        "local_dir_use_symlinks": False,
    }
    token = settings.HF_TOKEN or os.environ.get("HF_TOKEN")
    if token:
        download_kwargs["token"] = token
    try:
        snapshot_download(**download_kwargs)
    except Exception as exc:
        return "", str(exc)
    if config_path.exists():
        return str(target_dir), None
    return "", "Adapter files missing after download."


def _adapter_base_model(adapter_dir: Path) -> Optional[str]:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("base_model_name_or_path") or data.get("base_model")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Department model to use (e.g., finance, hr, engineering, it_support)")
    args, unknown = parser.parse_known_args()
    
    # Update SERVER_DEFAULTS if model specified via command line
    if args.model:
        candidate = _adapter_root() / args.model.lower()
        if candidate.is_dir():
            SERVER_DEFAULTS["peft_dir"] = str(candidate)
        else:
            print(f"Warning: Model directory '{candidate}' not found. Using default.")

    st.set_page_config(page_title="LoRA Chat", layout="wide")
    st.title("LoRA Chat")

    # GPU status
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    st.sidebar.subheader("System Info")
    st.sidebar.write(f"CUDA Available: {gpu_available}")
    st.sidebar.write(f"GPU Count: {gpu_count}")
    if gpu_available:
        st.sidebar.write(f"Current GPU: {torch.cuda.get_device_name(0)}")

    defaults = _resolve_defaults()

    # Model selection in sidebar
    available_models = _get_available_models()
    model_options: List[str] = ["Base Model"]
    model_identifiers: List[str] = [""]
    model_departments: List[Optional[str]] = [None]
    model_remote: List[bool] = [False]
    for identifier, display, is_remote, department in available_models:
        model_options.append(display)
        model_identifiers.append(identifier)
        model_departments.append(department)
        model_remote.append(is_remote)
    default_idx = 0
    for idx in range(1, len(model_options)):
        department = model_departments[idx]
        if not department:
            continue
        expected_dir = _adapter_root() / department
        current_dir = defaults["peft_dir"]
        if current_dir and Path(current_dir).resolve() == expected_dir.resolve():
            default_idx = idx
            break
    selected_model_display = st.sidebar.selectbox(
        "Select Department Model",
        model_options,
        index=default_idx,
        help="Choose which department-specific LoRA model to use for responses",
    )
    selected_idx = model_options.index(selected_model_display)
    selected_identifier = model_identifiers[selected_idx]
    selected_department_key = model_departments[selected_idx]
    selected_is_remote = model_remote[selected_idx]
    selected_peft_dir = ""
    selected_error = None
    if selected_identifier and selected_department_key:
        selected_peft_dir, selected_error = _ensure_adapter(selected_identifier, selected_is_remote, selected_department_key)
    if st.sidebar.button("Reload Model"):
        reset_cache_for(selected_peft_dir or None)
        st.session_state.pop("current_model", None)
        st.session_state.messages = []
        st.rerun()
    defaults["peft_dir"] = selected_peft_dir
    if selected_error:
        st.sidebar.error(f"Failed to prepare adapter: {selected_error}")
    resolved_base_model = SERVER_DEFAULTS["base_model"]
    override_base = settings.ADAPTER_BASE_MODELS.get(selected_department_key) if selected_department_key else None
    if override_base:
        resolved_base_model = override_base
    selected_path = Path(selected_peft_dir) if selected_peft_dir else None
    if selected_path and selected_path.is_dir():
        adapter_base = _adapter_base_model(selected_path)
        if adapter_base:
            resolved_base_model = adapter_base
        prefix = " remote" if selected_is_remote else ""
        st.sidebar.success(f"Using{prefix}: {selected_model_display}")
    else:
        if selected_identifier:
            st.sidebar.warning("Adapter not available; using base model.")
        else:
            st.sidebar.info("Using base model")
    defaults["base_model"] = resolved_base_model
    if not available_models:
        st.sidebar.warning("No department adapters detected")

    # Initialize session state for conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load model once (or reload if model changed)
    model_key = f"{defaults['peft_dir']}_{defaults['base_model']}"
    if "current_model" not in st.session_state or st.session_state.current_model != model_key:
        try:
            _get_model_and_tokenizer(
                defaults["base_model"],
                defaults["peft_dir"],
                defaults["dtype"],
                defaults["device_map"]
            )
            st.session_state.current_model = model_key
            st.session_state.model_loaded = True
            # Clear conversation when switching models
            st.session_state.messages = []
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
            return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare conversation
                    messages = [ChatMessage(**msg) for msg in st.session_state.messages]
                    tokenizer = _get_model_and_tokenizer(
                        defaults["base_model"],
                        defaults["peft_dir"],
                        defaults["dtype"],
                        defaults["device_map"]
                    )[1]
                    full_prompt = _conversation_to_prompt(messages, tokenizer)

                    resolved_top_p = defaults["top_p"] if defaults["top_p"] < 1.0 else None
                    do_sample = defaults["temperature"] > 0 or (resolved_top_p is not None and resolved_top_p < 1.0)

                    result = generate_text(
                        full_prompt,
                        defaults["max_new_tokens"],
                        defaults["base_model"],
                        defaults["peft_dir"],
                        defaults["dtype"],
                        defaults["device_map"],
                        return_full_text=False,
                        temperature=defaults["temperature"],
                        top_p=resolved_top_p,
                        do_sample=do_sample,
                    )
                    response = result.text
                except Exception as exc:
                    response = f"Error: {exc}"

            st.markdown(response)
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
            if selected_department_key:
                append_chat_record(selected_department_key, prompt, response, metadata={"source": "streamlit"})


if __name__ == "__main__":
    main()
