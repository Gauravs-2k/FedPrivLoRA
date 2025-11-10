import flwr as fl

from app.federation.department_client import DepartmentLoraClient


def main() -> None:
    client = DepartmentLoraClient(department="sales")
    fl.client.start_client(server_address="localhost:8081", client=client.to_client())


if __name__ == "__main__":
    main()
