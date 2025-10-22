import os
import sys
from pathlib import Path
import argparse

import streamlit as st
import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from model.inference import (
    SERVER_DEFAULTS,
    ChatMessage,
    _conversation_to_prompt,
    _get_model_and_tokenizer,
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


def _get_available_models():
    """Get list of available department LoRA models"""
    import os
    models = []
    for item in os.listdir('.'):
        if item.startswith('qwen_dept_lora_') and os.path.isdir(item):
            dept_name = item.replace('qwen_dept_lora_', '').replace('_', ' ').title()
            models.append((item, f"{dept_name} Department"))
    return models


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Department model to use (e.g., finance, hr, engineering, it_support)")
    args, unknown = parser.parse_known_args()
    
    # Update SERVER_DEFAULTS if model specified via command line
    if args.model:
        model_dir = f"qwen_dept_lora_{args.model.lower()}"
        if os.path.isdir(model_dir):
            SERVER_DEFAULTS["peft_dir"] = model_dir
        else:
            print(f"Warning: Model directory '{model_dir}' not found. Using default.")

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
    if available_models:
        model_options = [model[1] for model in available_models]
        model_dirs = [model[0] for model in available_models]
        default_idx = 0
        for i, (dir_name, display_name) in enumerate(available_models):
            if dir_name == defaults["peft_dir"]:
                default_idx = i
                break
        
        selected_model_display = st.sidebar.selectbox(
            "Select Department Model",
            model_options,
            index=default_idx,
            help="Choose which department-specific LoRA model to use for responses"
        )
        
        # Find the corresponding directory
        selected_idx = model_options.index(selected_model_display)
        selected_peft_dir = model_dirs[selected_idx]
        
        # Update defaults with selected model
        defaults["peft_dir"] = selected_peft_dir
        st.sidebar.success(f"Using: {selected_model_display}")
    else:
        st.sidebar.warning("No department models found. Using default model.")

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


if __name__ == "__main__":
    main()
