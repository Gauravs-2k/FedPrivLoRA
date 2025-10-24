# DML-project
DML project on lora finetuning.

## Flower Federated Learning Setup

This project includes a basic setup for Federated Learning using Flower (FLWR) with a server and clients for mobile and laptop.

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Server

On your laptop or cloud server:

```bash
python server.py
```

The server will start on `0.0.0.0:8081` and wait for 2 clients.

### Running the Clients

#### Mobile Client

On your mobile device (assuming Python environment):

```bash
python mobile_client.py
```

Note: Update the `server_address` in `mobile_client.py` to the actual server IP, e.g., `"192.168.1.100:8081"`.

#### Laptop Client

On another laptop or the same machine:

```bash
python laptop_client.py
```

Both clients are configured to connect to `localhost:8081`. To run both on the same laptop, start the server first, then run each client in a separate terminal.

### Model

The clients use MobileNetV2 for 10 classes, trained on CIFAR-10 dataset.

### Notes

- Ensure the server is accessible from the clients.
- For mobile deployment, you may need to set up Python on mobile or use a compatible environment.
- The training is simplified to 1 epoch per round.

## LoRA Inference API

### Local Run

With the virtual environment activated:

```bash
pip install -r requirements.txt
python app/model/inference.py --serve --host 0.0.0.0 --port 8000
```

REST endpoints (rooted at `http://localhost:8000`):

- `POST /generate` – basic text generation. Body:

	```json
	{
		"prompt": "Hello, how are you?",
		"max_new_tokens": 128
	}
	```

- `POST /v1/chat/completions` – OpenAI-compatible chat completions. Minimal body:

	```json
	{
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Write a haiku about LoRA."}
		],
		"max_tokens": 120,
		"temperature": 0.7
	}
	```

Both endpoints accept optional overrides for `model`, `peft_dir`, `dtype`, `device_map`, and generation parameters. Set `HF_TOKEN` in the environment if the base model requires authentication.

### Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

The container launches the Streamlit UI on port `8501`, available at `http://localhost:8501`. The compose file mounts `app/lora/qwen_lora` so the container can access local adapters and persists a Hugging Face cache in `./huggingface-cache`.

To stop the container:

```bash
docker compose down
```

### Streamlit UI

Launch an interactive UI with:

```bash
streamlit run app/streamlit_app.py
```

**Model Selection Options:**

1. **Interactive Dropdown**: Use the sidebar dropdown to switch between available department models (Finance, IT Support, etc.)

2. **Command Line Selection**: Specify a model when launching:
```bash
streamlit run app/streamlit_app.py -- --model finance
streamlit run app/streamlit_app.py -- --model it_support
```

**Available Models:**
- ✅ **Finance Department** (`qwen_dept_lora_finance`) - **Fully trained and tested**
- ✅ **IT Support Department** (`qwen_dept_lora_it_support`) - **Fully trained**
- ✅ **HR Department** (`qwen_dept_lora_hr`) - **Fully trained**
- ✅ **Engineering Department** (`qwen_dept_lora_engineering`) - **Fully trained**

*All department models are now trained and ready for inference!*

### Synthetic Instruction Datasets

Generate department contexts and Bonito-based Q&A datasets for FedPrivLoRA:

```bash
python app/dataset/generate_datasets.py --context-dir app/dataset --output-dir app/dataset --samples 1000
```

Each department JSONL file is written as `DEPT_dept.jsonl`, includes the specified number of pairs, and prints sample outputs plus training commands.

## LoRA Training

**Quick Start:** Train all department models automatically:
```bash
python app/lora/lora.py

python app/lora/lora.py --dataset app/dataset/dept/IT_SUPPORT_dept.jsonl --peft-output-dir qwen_dept_lora_it_support

```

### Automatic Training for All Departments

Train separate LoRA models for all departments automatically with hardcoded parameters:

```bash
python app/lora/lora.py
```

This will automatically:
- Use Qwen/Qwen1.5-1.8B-Chat as the base model
- Detect all department files (`*_dept.jsonl`) in `app/dataset/dept`
- Train separate LoRA models for each department with optimized settings
- Save models as `qwen_dept_lora_hr`, `qwen_dept_lora_finance`, etc.

**Training Configuration (hardcoded):**
- Base Model: Qwen/Qwen1.5-1.8B-Chat
- Dataset: app/dataset/dept (auto-detects all department files)
- Training: 1 epoch, batch size 1, gradient accumulation 8
- LoRA: rank 8, alpha 16, max length 256
- Optimization: FP16, gradient checkpointing enabled

### Custom Training (Advanced)

For custom training parameters, you can still modify the script directly or use the original argument-based approach by editing the hardcoded values in `app/lora/lora.py`.

### Train Combined Model (All Departments)

For training a combined model across all departments (legacy approach):

```bash
python app/lora/train_combined.py  # Use separate script for combined training
```

**Note:** The main training script now automatically trains separate department models. Use the command above for combined training if needed.

### GPU Memory Management

If you encounter CUDA out of memory errors, use the GPU memory clearing utility:

```bash
# Quick memory clear (recommended)
python app/model/gpu_clean.py
```

**Memory Optimization Tips:**
- Call `clear_gpu_memory()` between model inferences
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation
- Use smaller batch sizes if memory is limited
- Restart Python process if memory fragmentation becomes severe

### Training Parameters

Key options:
- `--base-model`: any HF causal LM (Qwen/Qwen1.5-1.8B-Chat)
- `--dataset`: path to JSONL file or HF dataset
- `--text-field` and `--response-field`: dataset columns
- `--peft-output-dir`: where to save LoRA adapters
- `--lora-r`: LoRA rank (8-16 recommended)
- `--lora-alpha`: LoRA scaling (16-32 recommended)
- Training: `--num-train-epochs`, `--learning-rate`, `--max-length`

The script saves adapters and tokenizer to `--peft-output-dir`.

## Uploading Models to Hugging Face Hub

After training your LoRA models, you can upload them to Hugging Face Hub for sharing and deployment.

### Authentication

First, log in to Hugging Face:

```bash
./env/bin/hf auth login
```

This will prompt for your Hugging Face token (get it from https://huggingface.co/settings/tokens).

### Upload Models

Upload each trained model to a new repository:

```bash
# Upload Finance model
./env/bin/hf upload Gaurav2k/qwen-dept-lora-finance qwen_dept_lora_finance --repo-type model


