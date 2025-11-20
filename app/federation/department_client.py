import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

import flwr as fl
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from app.dataset.chat_logs import load_chat_records
from app.federation.clustering import cluster_aware_average_selected
from app.federation.lora_utils import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DEVICE_MAP,
    DEFAULT_DTYPE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LORA_TARGET_MODULES,
    apply_lora_state,
    average_ndarrays,
    clone_numpy_state,
    collect_lora_parameter_names,
    collect_lora_state,
    create_lora_model,
    export_lora_adapter,
    lora_state_to_ndarrays,
    ndarrays_to_lora_state,
)


DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_BATCH_SIZE = 1
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MAX_RECORDS = 128


def _format_training_example(record: Dict[str, object]) -> str:
    user = str(record.get("user", ""))
    assistant = str(record.get("assistant", ""))
    return f"<user>{user}</user>\n<assistant>{assistant}</assistant>"


class DepartmentLoraClient(fl.client.NumPyClient):
    def __init__(
        self,
        department: str,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        r: int = DEFAULT_LORA_R,
        alpha: int = DEFAULT_LORA_ALPHA,
        dropout: float = DEFAULT_LORA_DROPOUT,
        target_modules: Optional[Sequence[str]] = DEFAULT_LORA_TARGET_MODULES,
        dtype: str = DEFAULT_DTYPE,
        device_map: str = DEFAULT_DEVICE_MAP,
        batch_size: int = DEFAULT_BATCH_SIZE,
        local_epochs: int = DEFAULT_LOCAL_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_records: int = DEFAULT_MAX_RECORDS,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        export_dir: Optional[Path] = None,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[object] = None,
        data_path: Optional[Path] = None,
    ) -> None:
        self.department = department
        self.base_model = base_model
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.dtype = dtype
        self.device_map = device_map
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.max_records = max_records
        self.max_seq_length = max_seq_length
        self.export_dir = export_dir or (Path("results") / "client_exports" / department)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = data_path

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.model, self.tokenizer = create_lora_model(
                base_model,
                r,
                alpha,
                dropout,
                target_modules=target_modules,
                dtype=dtype,
                device_map=device_map,
            )

        if device_map.lower() == "cpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        else:
            self.device = next(self.model.parameters()).device

        self.tokenizer.model_max_length = max_seq_length
        self.lora_parameter_names = collect_lora_parameter_names(self.model)
        self.training_dir = Path("results") / "client_runs" / department
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.model.train()

    def get_properties(self, config: Dict[str, fl.common.Scalar]) -> Dict[str, fl.common.Scalar]:
        return {"department": self.department}

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        state = collect_lora_state(self.model, self.lora_parameter_names)
        return lora_state_to_ndarrays(state, self.lora_parameter_names)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        if parameters:
            state = ndarrays_to_lora_state(self.lora_parameter_names, parameters, self.device)
            apply_lora_state(self.model, state)

        local_epochs = int(config.get("local_epochs", self.local_epochs))
        learning_rate = float(config.get("learning_rate", self.learning_rate))

        if "max_records" in config:
            self.max_records = int(config["max_records"])
        if "max_seq_length" in config:
            self.max_seq_length = int(config["max_seq_length"])

        records = self._load_records()
        if not records:
            state = collect_lora_state(self.model, self.lora_parameter_names)
            arrays = lora_state_to_ndarrays(state, self.lora_parameter_names)
            metrics = {"department": self.department, "train_loss": 0.0}
            return arrays, 0, metrics

        dataset = self._build_dataset(records)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=str(self.training_dir),
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=local_epochs,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        train_output = trainer.train()
        train_metrics = train_output.metrics or {}
        loss = float(train_metrics.get("train_loss", train_output.training_loss))

        state = collect_lora_state(self.model, self.lora_parameter_names)
        arrays = lora_state_to_ndarrays(state, self.lora_parameter_names)

        export_lora_adapter(
            self.base_model,
            self.lora_parameter_names,
            arrays,
            self.export_dir,
            r=self.r,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=self.target_modules,
            dtype=self.dtype,
            device_map=self.device_map,
        )
        metrics = {"department": self.department, "train_loss": loss}
        return arrays, len(records), metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> tuple[float, int, Dict[str, fl.common.Scalar]]:
        if parameters:
            state = ndarrays_to_lora_state(self.lora_parameter_names, parameters, self.device)
            apply_lora_state(self.model, state)

        records = self._load_records()
        if not records:
            return 0.0, 0, {"department": self.department}

        dataset = self._build_dataset(records)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        loader = DataLoader(dataset, batch_size=self.batch_size)

        total_loss = 0.0
        total_tokens = 0
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                labels = batch["input_ids"].clone()
                outputs = self.model(**batch, labels=labels)
                loss = outputs.loss
                tokens = labels.numel()
                total_loss += loss.item() * tokens
                total_tokens += tokens

        self.model.train()
        avg_loss = total_loss / total_tokens if total_tokens else 0.0
        return avg_loss, len(records), {"department": self.department, "val_loss": avg_loss}

    def _build_dataset(self, records: List[Dict[str, object]]) -> Dataset:
        texts = [_format_training_example(record) for record in records]
        dataset = Dataset.from_dict({"text": texts})

        def tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
            encodings = self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
            )
            return encodings

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        return tokenized

    def _load_records(self) -> List[Dict[str, object]]:
        if self.data_path:
            return self._load_jsonl_records(self.data_path)
        return load_chat_records(self.department, self.max_records)

    def _load_jsonl_records(self, path: Path) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(records) >= self.max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user = str(data.get("text") or "").strip()
                assistant = str(data.get("response") or "").strip()
                if not user or not assistant:
                    continue
                records.append({"user": user, "assistant": assistant})
        return records


def simulate_sequential_training(
    departments: Sequence[str],
    rounds: int,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    r: int = DEFAULT_LORA_R,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: Optional[Sequence[str]] = DEFAULT_LORA_TARGET_MODULES,
    dtype: str = DEFAULT_DTYPE,
    device_map: str = DEFAULT_DEVICE_MAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_records: int = DEFAULT_MAX_RECORDS,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    global_mixing: float = 0.1,
    dataset_map: Optional[Mapping[str, Path]] = None,
    num_clusters: int = 2,
) -> None:
    shared_model, shared_tokenizer = create_lora_model(
        base_model,
        r,
        alpha,
        dropout,
        target_modules=target_modules,
        dtype=dtype,
        device_map=device_map,
    )
    names = collect_lora_parameter_names(shared_model)
    initial_state = lora_state_to_ndarrays(collect_lora_state(shared_model, names), names)

    department_states: Dict[str, List[np.ndarray]] = {
        dept: clone_numpy_state(initial_state) for dept in departments
    }

    # Indices for LoRA A matrices only (where we apply clustering/mixing)
    a_indices = [index for index, param_name in enumerate(names) if "lora_A" in param_name]

    adapter_root = Path("results") / "adapters"
    adapter_root.mkdir(parents=True, exist_ok=True)

    for round_index in range(1, rounds + 1):
        round_weights: List[tuple[str, int]] = []

        for department in departments:
            params = clone_numpy_state(department_states[department])
            client = DepartmentLoraClient(
                department=department,
                base_model=base_model,
                r=r,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
                dtype=dtype,
                device_map=device_map,
                batch_size=batch_size,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                max_records=max_records,
                max_seq_length=max_seq_length,
                export_dir=Path("results") / "client_exports" / department,
                model=shared_model,
                tokenizer=shared_tokenizer,
                data_path=dataset_map.get(department) if dataset_map else None,
            )
            arrays, num_examples, _ = client.fit(
                params,
                {
                    "round": round_index,
                    "local_epochs": local_epochs,
                    "learning_rate": learning_rate,
                },
            )
            department_states[department] = clone_numpy_state(arrays)
            round_weights.append((department, num_examples if num_examples > 0 else 1))

        if round_weights:
            aggregated_inputs = [
                (department_states[dept], weight) for dept, weight in round_weights
            ]
            aggregated_per_item, _cluster_labels = cluster_aware_average_selected(
                aggregated_inputs,
                a_indices,
                num_clusters=num_clusters,
                max_dim=4096,
                max_iter=25,
                random_state=round_index,
            )

            # Mix cluster-level LoRA-A back into each department
            for (department, _), agg_a in zip(round_weights, aggregated_per_item):
                state = department_states[department]
                if agg_a:
                    if global_mixing > 0.0 and len(round_weights) > 1:
                        mix = float(global_mixing)
                        for position, index in enumerate(a_indices):
                            local_arr = state[index]
                            blended = (1.0 - mix) * local_arr + mix * agg_a[position]
                            state[index] = blended.astype(np.float32, copy=False)
                    else:
                        for position, index in enumerate(a_indices):
                            state[index] = np.array(agg_a[position], copy=True)
                department_states[department] = state

            # Export personalized adapters (per department, post-cluster sharing)
            for department in departments:
                export_lora_adapter(
                    base_model,
                    names,
                    department_states[department],
                    adapter_root / department,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    target_modules=target_modules,
                    dtype=dtype,
                    device_map=device_map,
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", nargs="+")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--global-mix", type=float, default=0.1)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device-map", default=DEFAULT_DEVICE_MAP)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--personal-root", type=Path)
    parser.add_argument("--num-clusters", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dataset_map: Optional[Dict[str, Path]] = None
    participants: Optional[List[str]] = None

    if args.personal_root:
        root = Path(args.personal_root)
        files = sorted(root.glob("*.jsonl"))
        if files:
            participants = [path.stem for path in files]
            dataset_map = {path.stem: path for path in files}

    if participants is None and args.simulate:
        participants = list(args.simulate)

    if participants:
        simulate_sequential_training(
            participants,
            args.rounds,
            global_mixing=args.global_mix,
            learning_rate=args.learning_rate,
            local_epochs=args.local_epochs,
            max_records=args.max_records,
            max_seq_length=args.max_seq_length,
            base_model=args.base_model,
            device_map=args.device_map,
            dtype=args.dtype,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            batch_size=args.batch_size,
            dataset_map=dataset_map,
            num_clusters=args.num_clusters,
        )

# Example:
# PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 -m app.federation.department_client \
#   --personal-root app/dataset/personal_clients \
#   --rounds 3 \
#   --global-mix 0.2 \
#   --max-records 128 \
#   --max-seq-length 256 \
#   --learning-rate 0.0002 \
#   --local-epochs 1 \
#   --batch-size 1 \
#   --device-map auto \
#   --num-clusters 3


# mac command:
    # python3 -m app.federation.department_client \
#   --personal-root app/dataset/personal_clients \
#   --rounds 3 \
#   --global-mix 0.2 \
#   --max-records 128 \
#   --max-seq-length 256 \
#   --learning-rate 0.0002 \
#   --local-epochs 1 \
#   --batch-size 1 \
#   --device-map auto \
#   --num-clusters 3



