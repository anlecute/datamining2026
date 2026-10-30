"""Create a compact cleaned JSONL file containing only project-required fields."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "cleaned.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "cleaned_minimal.jsonl"
DEFAULT_STATS = PROJECT_ROOT / "data" / "interim" / "cleaned_minimal_stats.json"
FIELDS = ("rating", "title", "text", "sentiment_class")
VALID_LABELS = {"Negative", "Neutral", "Positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only rating, title, text, and sentiment_class from cleaned JSONL."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    return parser.parse_args()


def validate_record(record: dict[str, Any], line_number: int) -> None:
    missing = [field for field in FIELDS if field not in record]
    if missing:
        raise ValueError(f"Line {line_number} is missing required fields: {', '.join(missing)}")
    if not isinstance(record["rating"], (int, float)) or isinstance(record["rating"], bool):
        raise ValueError(f"Line {line_number} has an invalid rating.")
    if not isinstance(record["title"], str) or not isinstance(record["text"], str):
        raise ValueError(f"Line {line_number} must contain string title and text fields.")
    if record["sentiment_class"] not in VALID_LABELS:
        raise ValueError(f"Line {line_number} has an invalid sentiment_class.")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    label_counts: Counter[str] = Counter()
    records_written = 0

    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            validate_record(record, line_number)
            minimal_record = {field: record[field] for field in FIELDS}
            target.write(json.dumps(minimal_record, ensure_ascii=False) + "\n")
            label_counts[minimal_record["sentiment_class"]] += 1
            records_written += 1

    summary = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "fields_kept": list(FIELDS),
        "records_written": records_written,
        "label_counts": dict(sorted(label_counts.items())),
    }
    args.stats.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
