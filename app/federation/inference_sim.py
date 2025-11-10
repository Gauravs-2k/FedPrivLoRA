import argparse
import json
from pathlib import Path

from app.model.inference import SERVER_DEFAULTS, generate_text, reset_cache_for


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--departments", nargs="*")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-new-tokens", type=int, default=int(SERVER_DEFAULTS["max_new_tokens"]))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--base-model")
    parser.add_argument("--dtype", default=SERVER_DEFAULTS["dtype"])
    parser.add_argument("--device-map", default=SERVER_DEFAULTS["device_map"])
    parser.add_argument("--include-base", action="store_true")
    return parser.parse_args()


def _collect_departments(requested: list[str] | None) -> list[str]:
    root = Path("results") / "adapters"
    if requested:
        return [dept for dept in requested if (root / dept).is_dir()]
    if not root.exists():
        return []
    return sorted(child.name for child in root.iterdir() if child.is_dir())


def _resolve_base_model(departments: list[str], override: str | None) -> str:
    if override:
        return override
    for department in departments:
        config_path = Path("results") / "adapters" / department / "adapter_config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            value = data.get("base_model_name_or_path") or data.get("base_model")
            if value:
                return value
    return SERVER_DEFAULTS["base_model"]


def _run_sequence(
    departments: list[str],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    include_base: bool,
    base_model: str,
    dtype: str,
    device_map: str,
) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    targets: list[tuple[str, str]] = []
    if include_base:
        targets.append(("base", ""))
    for department in departments:
        adapter_dir = Path("results") / "adapters" / department
        if adapter_dir.is_dir():
            targets.append((department, str(adapter_dir)))
    resolved_top_p = top_p if top_p < 1.0 else None
    do_sample = temperature > 0.0 or resolved_top_p is not None
    for label, adapter in targets:
        reset_cache_for(adapter or None)
        use_temperature = temperature if do_sample else None
        use_top_p = resolved_top_p if do_sample else None
        result = generate_text(
            prompt,
            max_new_tokens,
            base_model,
            adapter,
            dtype,
            device_map,
            return_full_text=False,
            temperature=use_temperature,
            top_p=use_top_p,
            do_sample=do_sample,
        )
        results.append((label, result.text))
    return results


def main() -> None:
    args = _args()
    departments = _collect_departments(args.departments)
    base_model = _resolve_base_model(departments, args.base_model)
    outputs = _run_sequence(
        departments,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.include_base,
        base_model,
        args.dtype,
        args.device_map,
    )
    for label, text in outputs:
        print(f"[{label}] {text}\n")


if __name__ == "__main__":
    main()
