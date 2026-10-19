"""Summarize mistakes from the held-out predictions of the selected model."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Negative", "Neutral", "Positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create error-analysis artifacts from test predictions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs" / "metrics" / "test_predictions_best_model.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "metrics")
    parser.add_argument("--examples-per-pair", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def markdown_report(summary: dict[str, object]) -> str:
    pairs = summary["misclassification_pairs"]
    pair_rows = "\n".join(
        f"| {row['actual_label']} | {row['predicted_label']} | {row['count']:,} |"
        for row in pairs
    )
    return f"""# Error analysis — mô hình tốt nhất

## Tóm tắt

| Chỉ số | Giá trị |
|---|---:|
| Tổng review test | {summary['total_test_records']:,} |
| Dự đoán đúng | {summary['correct_predictions']:,} |
| Dự đoán sai | {summary['incorrect_predictions']:,} |
| Accuracy | {summary['accuracy']:.4f} |
| Error rate | {summary['error_rate']:.2f}% |

## Các cặp nhầm lẫn

| Nhãn thật | Model dự đoán | Số review |
|---|---|---:|
{pair_rows}

File `error_examples.csv` có tối đa {summary['examples_per_pair']} ví dụ cho
mỗi cặp nhầm lẫn. Các ví dụ được chọn ngẫu nhiên có seed cố định để việc phân
tích có thể tái lập.
"""


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {args.input}. Run train_models.py first."
        )
    if args.examples_per_pair < 1:
        raise ValueError("--examples-per-pair must be at least 1")

    rows: list[dict[str, str]] = []
    with args.input.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        expected = {"actual_label", "predicted_label", "is_correct", "original_rating", "review_text"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError("The input CSV does not have the expected prediction columns.")
        for row in reader:
            if row["actual_label"] not in LABELS or row["predicted_label"] not in LABELS:
                raise ValueError("Prediction CSV contains an unknown label.")
            try:
                rating = float(row["original_rating"])
            except (TypeError, ValueError) as error:
                raise ValueError("Prediction CSV contains an invalid original rating.") from error
            if rating not in (1.0, 2.0, 3.0, 4.0, 5.0):
                raise ValueError("Prediction CSV contains an invalid original rating.")
            rows.append(row)

    if not rows:
        raise ValueError("Prediction CSV has no rows.")

    total = len(rows)
    incorrect = [row for row in rows if row["actual_label"] != row["predicted_label"]]
    pair_counts = Counter((row["actual_label"], row["predicted_label"]) for row in incorrect)
    pair_rows = [
        {"actual_label": actual, "predicted_label": predicted, "count": count}
        for (actual, predicted), count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    summary = {
        "input_file": str(args.input),
        "total_test_records": total,
        "correct_predictions": total - len(incorrect),
        "incorrect_predictions": len(incorrect),
        "accuracy": round((total - len(incorrect)) / total, 6),
        "error_rate": round(len(incorrect) / total * 100, 4),
        "misclassification_pairs": pair_rows,
        "examples_per_pair": args.examples_per_pair,
        "random_state": args.random_state,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "error_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "error_analysis_report.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in incorrect:
        grouped[(row["actual_label"], row["predicted_label"])].append(row)
    rng = random.Random(args.random_state)
    examples: list[dict[str, str]] = []
    for pair in sorted(grouped):
        candidates = grouped[pair]
        examples.extend(rng.sample(candidates, min(args.examples_per_pair, len(candidates))))
    with (args.output_dir / "error_examples.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["actual_label", "predicted_label", "original_rating", "review_text"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "actual_label": row["actual_label"],
                "predicted_label": row["predicted_label"],
                "original_rating": row["original_rating"],
                "review_text": row["review_text"],
            }
            for row in examples
        )

    print(f"Analysed {total:,} held-out predictions.")
    print(f"Error analysis: {args.output_dir / 'error_analysis_report.md'}")


if __name__ == "__main__":
    main()
