import json
import time
from pathlib import Path
from typing import Dict, List, Optional

CHAT_LOG_ROOT = Path(__file__).resolve().parents[2] / "results" / "chat_logs"


def append_chat_record(
    department: str,
    user_message: str,
    assistant_message: str,
    *,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    CHAT_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    record = {
        "department": department,
        "timestamp": time.time(),
        "user": user_message,
        "assistant": assistant_message,
    }
    if metadata:
        record.update(metadata)
    path = CHAT_LOG_ROOT / f"{department}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chat_records(department: str, max_records: Optional[int] = None) -> List[Dict[str, object]]:
    path = CHAT_LOG_ROOT / f"{department}.jsonl"
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
    if max_records is not None and max_records > 0:
        return records[-max_records:]
    return records


def list_recorded_departments() -> List[str]:
    if not CHAT_LOG_ROOT.exists():
        return []
    departments: List[str] = []
    for item in CHAT_LOG_ROOT.glob("*.jsonl"):
        departments.append(item.stem)
    return sorted(departments)
