import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

from app.utils.config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--personal-data", type=Path, default=base_dir / "personalised_data.json")
    parser.add_argument("--output", type=Path, default=base_dir / "edge_lora_engineering.jsonl")
    parser.add_argument("--personal-limit", type=int)
    parser.add_argument("--engineering-limit", type=int, default=1000)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--hf-token")
    parser.add_argument("--personal-output-dir", type=Path)
    parser.add_argument("--per-client-engineering-limit", type=int, default=0)
    return parser.parse_args()


def load_personal_pairs(path: Path) -> Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing personal data file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: List[Dict[str, str]] = []
    per_client: Dict[str, List[Dict[str, str]]] = {}
    for entry in payload:
        messages = entry.get("messages") or []
        context_lines: List[str] = []
        user_id = str(entry.get("user_id") or "unknown")
        client_records = per_client.setdefault(user_id, [])
        for message in messages:
            content = (message.get("content") or "").strip()
            if not content:
                continue
            role_raw = (message.get("role") or "").strip()
            role = role_raw.lower()
            if role == "assistant":
                if context_lines and context_lines[-1].startswith("User:"):
                    prompt = "\n".join(context_lines)
                    record = {"text": prompt, "response": content}
                    records.append(record)
                    client_records.append(record)
                context_lines.append(f"Assistant: {content}")
            elif role == "user":
                context_lines.append(f"User: {content}")
            elif role == "system":
                context_lines.append(f"System: {content}")
            else:
                label = role_raw or "Unknown"
                context_lines.append(f"{label}: {content}")
    return records, per_client


def extract_prompt(example: Dict[str, object]) -> str:
    for key in ("input", "prompt", "instruction"):
        value = example.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    messages = example.get("messages")
    if isinstance(messages, list):
        parts: List[str] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            content = (item.get("content") or "").strip()
            if not content:
                continue
            role = (item.get("role") or "").strip().lower()
            if role == "assistant":
                break
            label = "System" if role == "system" else "User"
            parts.append(f"{label}: {content}")
        return "\n".join(parts)
    return ""


def extract_response(example: Dict[str, object]) -> str:
    for key in ("output", "response", "completion"):
        value = example.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    messages = example.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, dict):
                continue
            content = (item.get("content") or "").strip()
            if not content:
                continue
            role = (item.get("role") or "").strip().lower()
            if role == "assistant":
                return content
    return ""


def load_engineering_samples(limit: int, token: Optional[str]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if limit <= 0:
        return records
    try:
        dataset = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True, token=token)
    except Exception as error:
        print(f"Failed to load OpenCodeInstruct: {error}")
        return records
    for example in dataset:
        if not isinstance(example, dict):
            continue
        prompt = extract_prompt(example)
        response = extract_response(example)
        if prompt and response:
            records.append({"text": prompt, "response": response})
        if len(records) >= limit:
            break
    return records


def deduplicate(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique: List[Dict[str, str]] = []
    for item in records:
        key = (item["text"], item["response"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def write_jsonl(path: Path, records: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sanitize_user_id(value: str) -> str:
    cleaned = [c if c.isalnum() or c in {"-", "_"} else "-" for c in value]
    result = "".join(cleaned).strip("-")
    return result or "client"


def write_personal_outputs(
    output_dir: Path,
    personal: Dict[str, List[Dict[str, str]]],
    limit: Optional[int],
    extras: Optional[Dict[str, List[Dict[str, str]]]] = None,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for user_id, items in personal.items():
        dataset = deduplicate(items)
        if limit and limit > 0:
            dataset = dataset[:limit]
        if extras and user_id in extras:
            dataset = deduplicate(dataset + extras[user_id])
        if not dataset:
            continue
        filename = sanitize_user_id(user_id) + ".jsonl"
        path = output_dir / filename
        write_jsonl(path, dataset)
        counts[str(path)] = len(dataset)
    return counts


def assign_engineering_records(
    engineering: List[Dict[str, str]],
    user_ids: List[str],
    per_client: int,
    seed: int,
) -> Dict[str, List[Dict[str, str]]]:
    if per_client <= 0 or not engineering or not user_ids:
        return {}
    pool = engineering.copy()
    rnd = random.Random(seed)
    rnd.shuffle(pool)
    assigned: Dict[str, List[Dict[str, str]]] = {}
    index = 0
    total = len(pool)
    for user_id in user_ids:
        chunk: List[Dict[str, str]] = []
        for _ in range(per_client):
            record = pool[index % total]
            chunk.append(record)
            index += 1
        assigned[user_id] = chunk
    return assigned


def main() -> None:
    args = parse_args()
    personal_records, personal_by_client = load_personal_pairs(args.personal_data)
    personal_records = deduplicate(personal_records)
    if args.personal_limit and args.personal_limit > 0:
        personal_records = personal_records[:args.personal_limit]
    token = args.hf_token or settings.HF_TOKEN
    engineering_records = load_engineering_samples(args.engineering_limit, token)
    combined = deduplicate(personal_records + engineering_records)
    random.Random(args.shuffle_seed).shuffle(combined)
    write_jsonl(args.output, combined)
    engineering_by_client = assign_engineering_records(
        engineering_records,
        list(personal_by_client.keys()),
        args.per_client_engineering_limit,
        args.shuffle_seed,
    )
    personal_outputs: Dict[str, int] = {}
    if args.personal_output_dir:
        personal_output_dir = args.personal_output_dir
        personal_output_dir.mkdir(parents=True, exist_ok=True)
        personal_outputs = write_personal_outputs(
            personal_output_dir,
            personal_by_client,
            args.personal_limit,
            engineering_by_client,
        )
    summary = {
        "personal_records": len(personal_records),
        "engineering_records": len(engineering_records),
        "combined_records": len(combined),
        "output": str(args.output),
    }
    if personal_outputs:
        summary["personal_files"] = personal_outputs
    if engineering_by_client:
        summary["per_client_engineering_limit"] = args.per_client_engineering_limit
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
