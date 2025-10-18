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
