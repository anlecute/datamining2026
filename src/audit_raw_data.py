"""Create reproducible evidence for the raw-data preprocessing decisions."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "All_Beauty.jsonl" / "All_Beauty.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "raw_data_audit.json"


def normalize_text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def label_from_rating(rating: float) -> str | None:
    if rating in (1.0, 2.0):
        return "Negative"
    if rating == 3.0:
        return "Neutral"
    if rating in (4.0, 5.0):
        return "Positive"
    return None


def exact_record_hash(record: dict[str, Any]) -> bytes:
    canonical = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(canonical.encode("utf-8")).digest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit every raw JSONL record using the preprocessing rules."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Raw JSONL file was not found: {args.input}")

    counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    seen_records: set[bytes] = set()

    with args.input.open("r", encoding="utf-8") as source:
        for line in source:
            counts["total_lines"] += 1
            if not line.strip():
                counts["blank_lines"] += 1
                continue

            counts["non_blank_lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["invalid_json_or_non_object_records"] += 1
                continue
            if not isinstance(record, dict):
                counts["invalid_json_or_non_object_records"] += 1
                continue

            try:
                rating = float(record.get("rating"))
            except (TypeError, ValueError):
                counts["invalid_rating_records"] += 1
                continue
            label = label_from_rating(rating)
            if label is None:
                counts["invalid_rating_records"] += 1
                continue

            title = normalize_text(record.get("title"))
            text = normalize_text(record.get("text"))
            if not title and not text:
                counts["empty_title_and_text_records"] += 1
                continue

            record_hash = exact_record_hash(record)
            if record_hash in seen_records:
                counts["exact_duplicate_records"] += 1
                continue
            seen_records.add(record_hash)
            counts["accepted_records"] += 1
            label_counts[label] += 1

    for key in (
        "blank_lines",
        "invalid_json_or_non_object_records",
        "invalid_rating_records",
        "empty_title_and_text_records",
        "exact_duplicate_records",
        "accepted_records",
    ):
        counts.setdefault(key, 0)

    audit = {
        "input_file": str(args.input),
        "input_file_size_bytes": args.input.stat().st_size,
        "input_file_sha256": file_sha256(args.input),
        "rules_checked_in_order": [
            "Blank line",
            "Valid JSON object",
            "Rating is one of 1, 2, 3, 4, 5",
            "Title and text are not both empty after whitespace normalization",
            "Exact duplicate JSON record",
        ],
        "counts": dict(counts),
        "label_counts_after_all_rules": dict(sorted(label_counts.items())),
        "accounting_check": (
            counts["blank_lines"]
            + counts["invalid_json_or_non_object_records"]
            + counts["invalid_rating_records"]
            + counts["empty_title_and_text_records"]
            + counts["exact_duplicate_records"]
            + counts["accepted_records"]
            == counts["total_lines"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
