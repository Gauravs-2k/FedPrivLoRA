import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from distilabel.models.llms import OpenAILLM

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.utils.config import settings

DEFAULT_MODEL = settings.LLM_MODEL
DEPARTMENTS = {
    "HR Department": "hr_context.txt",
    "Finance Department": "finance_context.txt",
    "Engineering Department": "engineering_context.txt",
    "IT Support Department": "it_support_context.txt",
}
DEFAULT_SAMPLE_COUNT = 100
OUTPUT_TEMPLATE = "{slug}_dept.jsonl"
DEFAULT_PREVIEW_COUNT = 3
SAMPLING_KWARGS = {
    "max_completion_tokens": 1024,
    "top_p": 0.95,
    "temperature": 0.7,
}
JSON_PATTERN = re.compile(r"{[\s\S]*}")


def normalize_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def read_contexts(base_dir: Path) -> Dict[str, str]:
    contexts: Dict[str, str] = {}
    for name, filename in DEPARTMENTS.items():
        path = base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing context file: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"empty context file: {path}")
        contexts[name] = text
    return contexts


def build_prompt(department: str, context: str, iteration: int = 0) -> List[Dict[str, str]]:
    system_prompt = (
        "You create diverse instruction-tuning question and answer pairs for enterprise teams."
        " Generate UNIQUE questions covering different processes, policies, and scenarios."
        " Avoid repetitive or similar questions."
        " Return strict JSON objects with keys 'question' and 'answer'."
    )
    
    # Add variety instructions based on iteration
    variety_hints = [
        "Focus on processes and workflows.",
        "Focus on policies and guidelines.",
        "Focus on troubleshooting and problem-solving.",
        "Focus on requests and approvals.",
        "Focus on best practices and standards.",
        "Focus on tools and systems.",
        "Focus on compliance and security.",
        "Focus on common issues and solutions.",
    ]
    
    hint = variety_hints[iteration % len(variety_hints)]
    
    user_prompt = (
        f"Department: {department}\n"
        f"Context:\n{context}\n\n"
        f"Generate exactly ONE UNIQUE question about the department and a concise answer.\n"
        f"{hint}\n"  # Add variety hint
        "Ensure the question is different from typical onboarding/basic questions."
        " Return valid JSON: {\"question\": \"...\", \"answer\": \"...\"}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]



def parse_generation(text: str) -> Optional[Tuple[str, str]]:
    candidate = text.strip()
    # Remove markdown code blocks
    if candidate.startswith("```json"):
        candidate = candidate[7:].strip()
    if candidate.endswith("```"):
        candidate = candidate[:-3].strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        match = JSON_PATTERN.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    question = str(data.get("question", "")).strip()
    answer = str(data.get("answer", "")).strip()
    if not question or not answer:
        return None
    return question, answer


def generate_pairs(llm: OpenAILLM, department: str, context: str, count: int, batch_size: int = 10) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    
    total_attempts = 0
    max_attempts = max(count * 3, count + 1)
    iteration = 0  # Track iterations for variety
    
    while len(pairs) < count and total_attempts < max_attempts:
        remaining = count - len(pairs)
        num_to_generate = min(batch_size, remaining)
        attempts = 0
        batch_pairs = []
        
        while len(batch_pairs) < num_to_generate and attempts < max_attempts:
            attempts += 1
            total_attempts += 1
            
            prompt = build_prompt(department, context, iteration)  # Add iteration
            iteration += 1
            
            outputs = llm.generate(
                inputs=[prompt],
                num_generations=1,
                temperature=SAMPLING_KWARGS["temperature"],
                top_p=SAMPLING_KWARGS["top_p"],
            )
            if not outputs or not outputs[0]["generations"]:
                continue
            text = outputs[0]["generations"][0] or ""
            parsed = parse_generation(text)
            if not parsed or parsed in seen:
                continue
            seen.add(parsed)
            batch_pairs.append(parsed)
        pairs.extend(batch_pairs)
    return pairs


def write_jsonl_batch(path: Path, pairs: List[Tuple[str, str]], mode: str = "a") -> None:
    with path.open(mode, encoding="utf-8") as fp:
        for question, answer in pairs:
            fp.write(json.dumps({"text": question, "response": answer}, ensure_ascii=False))
            fp.write("\n")


def print_samples(name: str, pairs: List[Tuple[str, str]], limit: int) -> None:
    print(f"Sample outputs for {name}:")
    if not pairs:
        print("  (none)")
        return
    for index, (question, answer) in enumerate(pairs[:max(limit, 0)], start=1):
        print(f"  [{index}] Q: {question}")
        print(f"      A: {answer}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--preview", type=int, default=DEFAULT_PREVIEW_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_dir = args.context_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts = read_contexts(context_dir)
    llm = OpenAILLM(
        model=args.model,
        base_url=settings.LLM_MODEL_ENDPOINT,
        api_key=settings.LLM_API_KEY,
    )
    llm.load()
    stats: Dict[str, int] = {}
    batch_size = 10  # Generate in batches of 10
    try:
        for name, context in contexts.items():
            slug = normalize_slug(name).replace("_department", "").upper()
            output_path = output_dir / OUTPUT_TEMPLATE.format(slug=slug)
            print(f"Generating data for {name} -> {output_path.name}")
            total_pairs = 0
            # Remove existing file if it exists
            if output_path.exists():
                output_path.unlink()
            while total_pairs < args.samples:
                remaining = args.samples - total_pairs
                num_to_gen = min(batch_size, remaining)
                pairs = generate_pairs(llm, name, context, num_to_gen, batch_size)
                if pairs:
                    write_jsonl_batch(output_path, pairs, "a")
                    total_pairs += len(pairs)
                    print(f"  generated {len(pairs)} pairs, total: {total_pairs}/{args.samples}")
                else:
                    print(f"  no new pairs generated, stopping for {name}")
                    break
            if total_pairs < args.samples:
                print(f"  warning: only {total_pairs} pairs generated; expected {args.samples}")
            stats[name] = total_pairs
            print(f"  saved {total_pairs} pairs to {output_path}")
            # Show a few samples at the end
            if total_pairs > 0:
                # Read back the last few pairs for preview
                with output_path.open("r", encoding="utf-8") as fp:
                    lines = fp.readlines()[-args.preview:]
                sample_pairs = []
                for line in lines:
                    data = json.loads(line.strip())
                    sample_pairs.append((data["text"], data["response"]))
                print_samples(name, sample_pairs, len(sample_pairs))
    finally:
        llm.unload()
    print("\nGeneration summary:")
    for name, count in stats.items():
        print(f"{name}: {count} pairs")


if __name__ == "__main__":
    main()
