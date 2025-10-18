import flwr as fl

# Simple FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,  # Wait for both mobile and laptop
    min_fit_clients=2,
    min_evaluate_clients=2,
)

fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)