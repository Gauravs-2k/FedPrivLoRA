import os
import sys
from pathlib import Path

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


def main():
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

    # Initialize session state for conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load model once
    if "model_loaded" not in st.session_state:
        try:
            _get_model_and_tokenizer(
                defaults["base_model"],
                defaults["peft_dir"],
                defaults["dtype"],
                defaults["device_map"]
            )
            st.session_state.model_loaded = True
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
