"""Train reproducible TF-IDF sentiment baselines without data leakage."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEANED = PROJECT_ROOT / "data" / "interim" / "cleaned.jsonl"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
LABELS = ["Negative", "Neutral", "Positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline sentiment classifiers.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument(
        "--max-records",
        type=int,
        default=50_000,
        help="Uniform random sample size; use 0 to train on every cleaned record.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args()


def load_records(path: Path, max_records: int, seed: int) -> tuple[list[str], list[str], list[float], int]:
    """Read JSONL and optionally retain a uniform reservoir sample in one pass."""
    if not path.is_file():
        raise FileNotFoundError(f"Run preprocessing first; file not found: {path}")
    if max_records < 0:
        raise ValueError("--max-records must be 0 or a positive integer")

    rng = random.Random(seed)
    sampled: list[tuple[str, str, float]] = []
    total_valid = 0
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            text, label = record.get("review_text"), record.get("sentiment_class")
            try:
                rating = float(record.get("rating"))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid rating in cleaned record at line {line_number}") from error
            if not isinstance(text, str) or label not in LABELS or rating not in (1.0, 2.0, 3.0, 4.0, 5.0):
                raise ValueError(f"Invalid cleaned record at line {line_number}")
            total_valid += 1
            if max_records == 0:
                sampled.append((text, label, rating))
            elif len(sampled) < max_records:
                sampled.append((text, label, rating))
            else:
                replacement_index = rng.randrange(total_valid)
                if replacement_index < max_records:
                    sampled[replacement_index] = (text, label, rating)

    if not sampled:
        raise ValueError("No cleaned records were found.")
    texts, labels, ratings = zip(*sampled)
    print(f"Loaded {len(sampled):,} records (from {total_valid:,} cleaned records).")
    return list(texts), list(labels), list(ratings), total_valid


def write_metrics(rows: list[dict[str, object]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "model_comparison.json"
    csv_path = directory / "model_comparison.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_experiment_configuration(
    args: argparse.Namespace,
    total_cleaned_records: int,
    sampled_labels: list[str],
    train_labels: list[str],
    test_labels: list[str],
    directory: Path,
) -> None:
    """Save the exact modelling protocol used for reproducible reporting."""
    directory.mkdir(parents=True, exist_ok=True)
    configuration = {
        "input_file": str(args.input),
        "total_cleaned_records": total_cleaned_records,
        "sample_size": len(sampled_labels),
        "sampling_method": "Uniform reservoir sampling; 0 means every cleaned record.",
        "random_state": args.random_state,
        "split": {
            "method": "Stratified train/test split",
            "train_fraction": 1 - args.test_size,
            "test_fraction": args.test_size,
            "train_records": len(train_labels),
            "test_records": len(test_labels),
            "sample_class_counts": dict(sorted(Counter(sampled_labels).items())),
            "train_class_counts": dict(sorted(Counter(train_labels).items())),
            "test_class_counts": dict(sorted(Counter(test_labels).items())),
        },
        "feature_extraction": {
            "method": "TF-IDF",
            "lowercase": True,
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "max_features": 20_000,
            "leakage_prevention": "Vectorizer is fit only on training text; test text is transform-only.",
        },
        "models": {
            "Dummy (most frequent)": {"strategy": "most_frequent"},
            "Multinomial Naive Bayes": {},
            "Logistic Regression": {"max_iter": 1000, "class_weight": "balanced"},
            "Linear SVM": {"class_weight": "balanced"},
        },
        "selection_metric": "Macro F1",
    }
    (directory / "experiment_configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_test_predictions(
    texts: list[str],
    actual_labels: list[str],
    predicted_labels: list[str],
    ratings: list[float],
    directory: Path,
) -> None:
    """Save held-out predictions for transparent evaluation and error analysis."""
    directory.mkdir(parents=True, exist_ok=True)
    prediction_path = directory / "test_predictions_best_model.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["actual_label", "predicted_label", "is_correct", "original_rating", "review_text"],
        )
        writer.writeheader()
        for text, actual, predicted, rating in zip(texts, actual_labels, predicted_labels, ratings):
            writer.writerow(
                {
                    "actual_label": actual,
                    "predicted_label": predicted,
                    "is_correct": actual == predicted,
                    "original_rating": rating,
                    "review_text": text,
                }
            )


def save_confusion_matrix(y_true: list[str], y_pred: list[str], model_name: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS).plot(
        ax=axis, colorbar=False, cmap="Blues"
    )
    axis.set_title(f"Confusion matrix — {model_name}")
    figure.tight_layout()
    figure.savefig(directory / "confusion_matrix_best_model.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    texts, labels, ratings, total_cleaned_records = load_records(args.input, args.max_records, args.random_state)
    x_train, x_test, y_train, y_test, _, rating_test = train_test_split(
        texts,
        labels,
        ratings,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    # Fit only on training text: the held-out test vocabulary and IDF remain unseen.
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=20_000
    )
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)
    write_experiment_configuration(
        args,
        total_cleaned_records=total_cleaned_records,
        sampled_labels=labels,
        train_labels=y_train,
        test_labels=y_test,
        directory=args.metrics_dir,
    )

    models = {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced"),
    }
    rows: list[dict[str, object]] = []
    best_name = ""
    best_predictions: list[str] = []
    best_macro_f1 = -1.0

    for name, model in models.items():
        model.fit(x_train_tfidf, y_train)
        predictions = model.predict(x_test_tfidf)
        macro_f1 = f1_score(y_test, predictions, labels=LABELS, average="macro", zero_division=0)
        row = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, predictions), 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(
                f1_score(y_test, predictions, labels=LABELS, average="weighted", zero_division=0), 6
            ),
        }
        rows.append(row)
        print(f"{name}: accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}")
        if macro_f1 > best_macro_f1:
            best_name, best_predictions, best_macro_f1 = name, list(predictions), macro_f1

    write_metrics(rows, args.metrics_dir)
    report = classification_report(y_test, best_predictions, labels=LABELS, zero_division=0)
    (args.metrics_dir / "best_model_report.txt").write_text(
        f"Best model: {best_name}\n\n{report}", encoding="utf-8"
    )
    write_test_predictions(x_test, y_test, best_predictions, rating_test, args.metrics_dir)
    save_confusion_matrix(y_test, best_predictions, best_name, args.figures_dir)
    print(f"Best model by macro F1: {best_name}")


if __name__ == "__main__":
    main()
