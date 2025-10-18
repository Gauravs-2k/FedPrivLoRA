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
python app/lora/lora_inference.py --serve --host 0.0.0.0 --port 8000
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
streamlit run app/lora/streamlit_app.py
```

The UI exposes model and adapter configuration fields, sampling controls, and displays token usage for each generation. The Docker Compose setup above provides the same UI inside a container.
