"""Create reproducible EDA tables and figures from cleaned Amazon reviews."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from math import ceil
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Negative", "Neutral", "Positive"]
RATINGS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create EDA artifacts from cleaned reviews.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "interim" / "cleaned.jsonl")
    parser.add_argument(
        "--preprocessing-stats",
        type=Path,
        default=ROOT / "data" / "interim" / "preprocessing_stats.json",
    )
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "outputs" / "metrics")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs" / "figures")
    return parser.parse_args()


def nearest_rank(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[ceil(len(ordered) * fraction) - 1]


def describe(values: list[int]) -> dict[str, int | float]:
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": nearest_rank(values, 0.95),
        "p99": nearest_rank(values, 0.99),
    }


def label_bars(axis: plt.Axes) -> None:
    for bar in axis.patches:
        height = bar.get_height()
        axis.annotate(
            f"{height:,.0f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def plot_bars(categories: list[str], values: list[int], colors: list[str], title: str, x_label: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(categories, values, color=colors)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Number of reviews")
    label_bars(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_length_histogram(lengths: list[int], path: Path) -> None:
    p99 = nearest_rank(lengths, 0.99)
    visible = [value for value in lengths if value <= p99]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(visible, bins=50, color="#72B7B2", edgecolor="white")
    axis.set_title("Review length distribution")
    axis.set_xlabel(f"Words per review (shown through 99th percentile = {p99})")
    axis.set_ylabel("Number of reviews")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_length_boxplot(lengths_by_label: dict[str, list[int]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot([lengths_by_label[label] for label in LABELS], tick_labels=LABELS, showfliers=False)
    axis.set_title("Review length by sentiment class")
    axis.set_xlabel("Sentiment class")
    axis.set_ylabel("Words per review")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_interpretation(summary: dict[str, object], path: Path) -> None:
    cleaning = summary["cleaning_summary"]
    labels = summary["label_distribution"]
    lengths = summary["review_length_words"]
    by_label = summary["review_length_by_sentiment"]
    dominant = max(LABELS, key=lambda label: labels[label]["count"])
    content = f"""# Kết quả EDA — All_Beauty

## Bảng 1. Tóm tắt làm sạch

| Chỉ số | Giá trị |
|---|---:|
| Record raw | {cleaning['raw_records']:,} |
| Record trùng hoàn toàn bị loại | {cleaning['duplicate_records_removed']:,} |
| Record sạch | {cleaning['cleaned_records']:,} |

Dataset có {cleaning['raw_records']:,} record raw và còn {cleaning['cleaned_records']:,}
record sau làm sạch. Chênh lệch là {cleaning['raw_records'] - cleaning['cleaned_records']:,}
record trùng hoàn toàn; số liệu chi tiết nằm trong `raw_data_audit.json`.

## Bảng 2. Phân bố sentiment class

| Lớp | Số review | Tỷ lệ |
|---|---:|---:|
| Negative | {labels['Negative']['count']:,} | {labels['Negative']['percentage']:.2f}% |
| Neutral | {labels['Neutral']['count']:,} | {labels['Neutral']['percentage']:.2f}% |
| Positive | {labels['Positive']['count']:,} | {labels['Positive']['percentage']:.2f}% |

Lớp {dominant} chiếm tỷ lệ cao nhất, trong khi Neutral là lớp ít nhất. Vì dữ
liệu mất cân bằng, Macro F1 phù hợp hơn Accuracy khi chọn model.

## Bảng 3. Độ dài review theo số từ

| Nhóm | Count | Mean | Median | Min | Max | P95 |
|---|---:|---:|---:|---:|---:|---:|
| Toàn bộ | {lengths['overall']['count']:,} | {lengths['overall']['mean']:.2f} | {lengths['overall']['median']:.1f} | {lengths['overall']['min']} | {lengths['overall']['max']} | {lengths['overall']['p95']} |
| Negative | {by_label['Negative']['count']:,} | {by_label['Negative']['mean']:.2f} | {by_label['Negative']['median']:.1f} | {by_label['Negative']['min']} | {by_label['Negative']['max']} | {by_label['Negative']['p95']} |
| Neutral | {by_label['Neutral']['count']:,} | {by_label['Neutral']['mean']:.2f} | {by_label['Neutral']['median']:.1f} | {by_label['Neutral']['min']} | {by_label['Neutral']['max']} | {by_label['Neutral']['p95']} |
| Positive | {by_label['Positive']['count']:,} | {by_label['Positive']['mean']:.2f} | {by_label['Positive']['median']:.1f} | {by_label['Positive']['min']} | {by_label['Positive']['max']} | {by_label['Positive']['p95']} |

Histogram chỉ hiển thị đến phân vị 99% để một vài review rất dài không làm che
phần lớn dữ liệu. Boxplot ẩn outlier để dễ so sánh nhóm chính giữa ba lớp.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Run preprocessing first; missing: {args.input}")

    ratings: Counter[int] = Counter()
    labels: Counter[str] = Counter()
    lengths: list[int] = []
    lengths_by_label: dict[str, list[int]] = defaultdict(list)

    with args.input.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            text = record.get("review_text")
            label = record.get("sentiment_class")
            try:
                rating = int(float(record.get("rating")))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid rating at cleaned line {line_number}") from error
            if not isinstance(text, str) or label not in LABELS or rating not in RATINGS:
                raise ValueError(f"Invalid cleaned record at line {line_number}")
            word_count = len(text.split())
            ratings[rating] += 1
            labels[label] += 1
            lengths.append(word_count)
            lengths_by_label[label].append(word_count)

    if not lengths:
        raise ValueError("No cleaned reviews found.")

    processing_counts: dict[str, int] = {}
    if args.preprocessing_stats.is_file():
        preprocessing = json.loads(args.preprocessing_stats.read_text(encoding="utf-8"))
        processing_counts = preprocessing.get("processing_counts", {})

    total = len(lengths)
    summary = {
        "source_file": str(args.input),
        "cleaning_summary": {
            "raw_records": processing_counts.get("raw_records", total),
            "duplicate_records_removed": processing_counts.get("duplicate_records_removed", 0),
            "cleaned_records": total,
        },
        "rating_distribution": {str(rating): ratings[rating] for rating in RATINGS},
        "label_distribution": {
            label: {"count": labels[label], "percentage": round(labels[label] / total * 100, 4)}
            for label in LABELS
        },
        "review_length_words": {"overall": describe(lengths)},
        "review_length_by_sentiment": {label: describe(lengths_by_label[label]) for label in LABELS},
    }

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    (args.metrics_dir / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_interpretation(summary, args.metrics_dir / "eda_interpretation.md")
    plot_bars(
        [str(rating) for rating in RATINGS],
        [ratings[rating] for rating in RATINGS],
        ["#4C78A8"] * len(RATINGS),
        "Distribution of ratings",
        "Rating (stars)",
        args.figures_dir / "eda_rating_distribution.png",
    )
    plot_bars(
        LABELS,
        [labels[label] for label in LABELS],
        ["#E45756", "#F2CF5B", "#54A24B"],
        "Distribution of sentiment classes",
        "Sentiment class",
        args.figures_dir / "eda_sentiment_distribution.png",
    )
    plot_length_histogram(lengths, args.figures_dir / "eda_review_length_distribution.png")
    plot_length_boxplot(lengths_by_label, args.figures_dir / "eda_review_length_by_sentiment.png")
    print(f"EDA completed for {total:,} cleaned reviews.")
    print(f"Summary: {args.metrics_dir / 'eda_summary.json'}")
    print(f"Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
