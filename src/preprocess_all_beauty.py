"""Clean Amazon Reviews JSONL and derive the three sentiment labels.

The raw input is never changed.  Each non-empty input line is treated as an
independent JSON record, so the script remains suitable for large JSONL files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "All_Beauty.jsonl" / "All_Beauty.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"


def normalize_text(value: Any) -> str:
    """Return whitespace-normalized text, or an empty string for non-text."""
    return " ".join(value.split()) if isinstance(value, str) else ""


def label_from_rating(rating: float) -> str | None:
    """Map the project's five star ratings to its three sentiment classes."""
    if rating in (1.0, 2.0):
        return "Negative"
    if rating == 3.0:
        return "Neutral"
    if rating in (4.0, 5.0):
        return "Positive"
    return None


def exact_record_hash(record: dict[str, Any]) -> bytes:
    """Hash the complete canonical JSON record for exact-duplicate detection."""
    canonical = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(canonical.encode("utf-8")).digest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Amazon Reviews JSONL and add review_text and sentiment_class."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="Raw JSONL review file."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_dir: Path = args.output_dir
    cleaned_path = output_dir / "cleaned.jsonl"
    stats_path = output_dir / "preprocessing_stats.json"

    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL file was not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    seen_records: set[bytes] = set()

    with input_path.open("r", encoding="utf-8") as source, cleaned_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                stats["blank_lines_removed"] += 1
                continue

            stats["raw_records"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json_removed"] += 1
                continue

            if not isinstance(record, dict):
                stats["invalid_json_removed"] += 1
                continue

            try:
                rating = float(record.get("rating"))
            except (TypeError, ValueError):
                stats["invalid_rating_removed"] += 1
                continue

            label = label_from_rating(rating)
            if label is None:
                stats["invalid_rating_removed"] += 1
                continue

            title = normalize_text(record.get("title"))
            text = normalize_text(record.get("text"))
            if not title and not text:
                stats["empty_title_and_text_removed"] += 1
                continue

            record_hash = exact_record_hash(record)
            if record_hash in seen_records:
                stats["duplicate_records_removed"] += 1
                continue
            seen_records.add(record_hash)

            cleaned_record = dict(record)
            cleaned_record["review_text"] = " ".join(part for part in (title, text) if part)
            cleaned_record["sentiment_class"] = label
            target.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")
            stats["cleaned_records"] += 1
            label_counts[label] += 1

    summary = {
        "input_file": str(input_path),
        "output_file": str(cleaned_path),
        "deduplication_rule": "Only exact duplicate JSON records are removed.",
        "processing_counts": dict(stats),
        "label_counts": dict(sorted(label_counts.items())),
    }
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
