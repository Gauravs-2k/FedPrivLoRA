from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import flwr as fl
import numpy as np
from flwr.common import FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from app.federation.lora_utils import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DEVICE_MAP,
    DEFAULT_DTYPE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LORA_TARGET_MODULES,
    average_ndarrays,
    clone_numpy_state,
    collect_lora_parameter_names,
    collect_lora_state,
    create_lora_model,
    export_lora_adapter,
    lora_state_to_ndarrays,
)


class DepartmentLoraStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        get_properties_timeout: Optional[float] = None,
        global_mixing: float = 0.0,
        fit_config: Optional[Dict[str, Scalar]] = None,
        base_model: str = DEFAULT_BASE_MODEL,
        r: int = DEFAULT_LORA_R,
        alpha: int = DEFAULT_LORA_ALPHA,
        dropout: float = DEFAULT_LORA_DROPOUT,
        target_modules: Optional[Sequence[str]] = DEFAULT_LORA_TARGET_MODULES,
        dtype: str = DEFAULT_DTYPE,
        device_map: str = DEFAULT_DEVICE_MAP,
    ) -> None:
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            min_evaluate_clients=0,
            accept_failures=True,
        )
        model, _ = create_lora_model(
            base_model,
            r,
            alpha,
            dropout,
            target_modules=target_modules,
            dtype=dtype,
            device_map=device_map,
        )
        model.to("cpu")
        self.lora_parameter_names = collect_lora_parameter_names(model)
        initial_state = collect_lora_state(model, self.lora_parameter_names)
        self.template_state = lora_state_to_ndarrays(initial_state, self.lora_parameter_names)
        self.department_states: Dict[str, List[np.ndarray]] = {}
        self.global_state: List[np.ndarray] = clone_numpy_state(self.template_state)
        self.client_departments: Dict[str, str] = {}
        self.get_properties_timeout = get_properties_timeout
        self.global_mixing = global_mixing
        self.fit_config = dict(fit_config) if fit_config is not None else {}
        self.base_model = base_model
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.dtype = dtype
        self.device_map = device_map
        self.adapter_export_dir = Path("results") / "adapters"
        self.adapter_export_dir.mkdir(parents=True, exist_ok=True)

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def _resolve_client_department(self, client: ClientProxy) -> str:
        cached = client.properties.get("department")
        if isinstance(cached, str):
            self.client_departments[client.cid] = cached
            return cached
        cached_mapping = self.client_departments.get(client.cid)
        if cached_mapping is not None:
            return cached_mapping
        response = client.get_properties(GetPropertiesIns(config={}), self.get_properties_timeout, None)
        department = response.properties.get("department")
        if not isinstance(department, str):
            raise ValueError("Client did not provide a department identifier")
        client.properties["department"] = department
        self.client_departments[client.cid] = department
        return department

    def _export_department_adapter(self, department: str) -> None:
        arrays = self.department_states.get(department)
        if arrays is None:
            return
        output_dir = self.adapter_export_dir / department
        export_lora_adapter(
            self.base_model,
            self.lora_parameter_names,
            arrays,
            output_dir,
            r=self.r,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=self.target_modules,
            dtype=self.dtype,
            device_map=self.device_map,
        )

    def _ensure_department_state(self, department: str) -> List[np.ndarray]:
        if department not in self.department_states:
            self.department_states[department] = clone_numpy_state(self.template_state)
            self._export_department_adapter(department)
        return self.department_states[department]

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = dict(self.fit_config)
        config["round"] = server_round
        config["base_model"] = self.base_model
        config["lora_r"] = self.r
        config["lora_alpha"] = self.alpha
        config["lora_dropout"] = self.dropout
        config["dtype"] = self.dtype
        config["device_map"] = self.device_map
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        instructions: List[Tuple[ClientProxy, FitIns]] = []
        for client in clients:
            department = self._resolve_client_department(client)
            state = self._ensure_department_state(department)
            fit_parameters = ndarrays_to_parameters(clone_numpy_state(state))
            client_config = dict(config)
            client_config["department"] = department
            instructions.append((client, FitIns(fit_parameters, client_config)))
        return instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        grouped_updates: Dict[str, List[Tuple[Sequence[np.ndarray], int]]] = defaultdict(list)
        department_weights: Dict[str, int] = defaultdict(int)
        metric_values: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for client, fit_res in results:
            department = fit_res.metrics.get("department")
            if not isinstance(department, str):
                department = self._resolve_client_department(client)
            arrays = parameters_to_ndarrays(fit_res.parameters)
            grouped_updates[department].append((arrays, fit_res.num_examples))
            department_weights[department] += fit_res.num_examples
            loss_metric = fit_res.metrics.get("train_loss")
            if isinstance(loss_metric, (float, int)):
                metric_values[f"{department}_train_loss"].append((fit_res.num_examples, float(loss_metric)))
            acc_metric = fit_res.metrics.get("train_accuracy")
            if isinstance(acc_metric, (float, int)):
                metric_values[f"{department}_train_accuracy"].append((fit_res.num_examples, float(acc_metric)))
        updated_departments: List[str] = []
        for department, items in grouped_updates.items():
            aggregated = average_ndarrays(items)
            self.department_states[department] = [np.array(arr, copy=True) for arr in aggregated]
            updated_departments.append(department)
            self._export_department_adapter(department)
        if updated_departments:
            global_state = average_ndarrays(
                [(self.department_states[dept], department_weights[dept]) for dept in updated_departments]
            )
            self.global_state = [np.array(arr, copy=True) for arr in global_state]
            if self.global_mixing > 0.0 and len(updated_departments) > 1:
                mix = float(self.global_mixing)
                for department in updated_departments:
                    mixed = []
                    for local_arr, global_arr in zip(self.department_states[department], self.global_state):
                        blended = (1.0 - mix) * local_arr + mix * global_arr
                        mixed.append(blended.astype(np.float32, copy=False))
                    self.department_states[department] = mixed
        metrics: Dict[str, Scalar] = {"aggregated_departments": len(updated_departments)}
        for department, weight in department_weights.items():
            metrics[f"{department}_examples"] = float(weight)
        for metric_name, pairs in metric_values.items():
            total_weight = sum(weight for weight, _ in pairs)
            if total_weight > 0:
                value = sum(weight * score for weight, score in pairs) / total_weight
                metrics[metric_name] = float(value)
        parameters_aggregated = ndarrays_to_parameters(clone_numpy_state(self.global_state))
        return parameters_aggregated, metrics


def main() -> None:
    strategy = DepartmentLoraStrategy(
        min_fit_clients=2,
        min_available_clients=2,
        global_mixing=0.1,
        fit_config={"local_epochs": 1, "learning_rate": 1e-3},
    )

    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
