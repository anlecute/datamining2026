"""Export the cleaned Amazon review data to ARFF and a reviewable CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEANED = PROJECT_ROOT / "data" / "interim" / "cleaned.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
VALID_LABELS = {"Negative", "Neutral", "Positive"}


def escape_arff(value: str) -> str:
    """Escape a value for a single-quoted ARFF string literal."""
    return (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cleaned reviews to ARFF and CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_dir: Path = args.output_dir
    arff_path = output_dir / "amazon_sentiment.arff"
    csv_path = output_dir / "amazon_sentiment.csv"

    if not input_path.is_file():
        raise FileNotFoundError(f"Run preprocessing first; file not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with input_path.open("r", encoding="utf-8") as source, arff_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as arff, csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        arff.write("@RELATION amazon_reviews_sentiment\n\n")
        arff.write("@ATTRIBUTE review_text STRING\n")
        arff.write("@ATTRIBUTE sentiment_class {Negative,Neutral,Positive}\n\n")
        arff.write("@DATA\n")
        writer = csv.DictWriter(csv_file, fieldnames=["review_text", "sentiment_class"])
        writer.writeheader()

        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            text = record.get("review_text")
            label = record.get("sentiment_class")
            if not isinstance(text, str) or label not in VALID_LABELS:
                raise ValueError(f"Invalid cleaned record at line {line_number}")
            arff.write(f"'{escape_arff(text)}',{label}\n")
            writer.writerow({"review_text": text, "sentiment_class": label})
            rows_written += 1

    print(f"Exported {rows_written:,} records to {arff_path} and {csv_path}")


if __name__ == "__main__":
    main()
