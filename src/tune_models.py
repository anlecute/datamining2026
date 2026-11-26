"""Run controlled TF-IDF and linear-model tuning without using the test set to choose a model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from train_models import LABELS, load_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "interim" / "cleaned.jsonl"
DEFAULT_METRICS_DIR = ROOT / "outputs" / "metrics"
DEFAULT_FIGURES_DIR = ROOT / "outputs" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune controlled TF-IDF linear-model experiments.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--max-records", type=int, default=50_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fraction of the outer training set reserved for validation.",
    )
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args()


def experiment_definitions() -> list[dict[str, object]]:
    """Each experiment changes one stated factor against its model family baseline."""
    return [
        {
            "id": "lr_baseline",
            "model": "Logistic Regression",
            "ngram_range": (1, 2),
            "max_features": 20_000,
            "class_weight": "balanced",
            "C": 1.0,
        },
        {
            "id": "lr_unigram",
            "model": "Logistic Regression",
            "ngram_range": (1, 1),
            "max_features": 20_000,
            "class_weight": "balanced",
            "C": 1.0,
        },
        {
            "id": "lr_no_class_weight",
            "model": "Logistic Regression",
            "ngram_range": (1, 2),
            "max_features": 20_000,
            "class_weight": None,
            "C": 1.0,
        },
        {
            "id": "lr_10k_features",
            "model": "Logistic Regression",
            "ngram_range": (1, 2),
            "max_features": 10_000,
            "class_weight": "balanced",
            "C": 1.0,
        },
        {
            "id": "lr_c_0_5",
            "model": "Logistic Regression",
            "ngram_range": (1, 2),
            "max_features": 20_000,
            "class_weight": "balanced",
            "C": 0.5,
        },
        {
            "id": "lr_c_2_0",
            "model": "Logistic Regression",
            "ngram_range": (1, 2),
            "max_features": 20_000,
            "class_weight": "balanced",
            "C": 2.0,
        },
        {
            "id": "svm_c_0_5",
            "model": "Linear SVM",
            "ngram_range": (1, 2),
            "max_features": 20_000,
            "class_weight": "balanced",
            "C": 0.5,
        },
        {
            "id": "svm_c_1_0",
            "model": "Linear SVM",
            "ngram_range": (1, 2),
            "max_features": 20_000,
            "class_weight": "balanced",
            "C": 1.0,
        },
        {
            "id": "svm_c_2_0",
            "model": "Linear SVM",
            "ngram_range": (1, 2),
            "max_features": 20_000,
            "class_weight": "balanced",
            "C": 2.0,
        },
    ]


def make_model(experiment: dict[str, object]):
    if experiment["model"] == "Logistic Regression":
        return LogisticRegression(
            C=float(experiment["C"]),
            max_iter=1000,
            class_weight=experiment["class_weight"],
        )
    return LinearSVC(C=float(experiment["C"]), class_weight=experiment["class_weight"])


def make_vectorizer(experiment: dict[str, object]) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=experiment["ngram_range"],
        min_df=2,
        max_df=0.95,
        max_features=int(experiment["max_features"]),
    )


def score(y_true: list[str], predictions: list[str]) -> dict[str, float]:
    result = {
        "accuracy": round(accuracy_score(y_true, predictions), 6),
        "macro_f1": round(f1_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0), 6),
        "weighted_f1": round(
            f1_score(y_true, predictions, labels=LABELS, average="weighted", zero_division=0), 6
        ),
    }
    for label in LABELS:
        key = f"{label.lower()}_f1"
        result[key] = round(
            f1_score(y_true, predictions, labels=[label], average="macro", zero_division=0), 6
        )
    return result


def evaluate(experiment: dict[str, object], x_fit: list[str], y_fit: list[str], x_eval: list[str], y_eval: list[str]):
    vectorizer = make_vectorizer(experiment)
    x_fit_tfidf = vectorizer.fit_transform(x_fit)
    x_eval_tfidf = vectorizer.transform(x_eval)
    model = make_model(experiment)
    model.fit(x_fit_tfidf, y_fit)
    predictions = list(model.predict(x_eval_tfidf))
    return score(y_eval, predictions), predictions


def write_rows(rows: list[dict[str, object]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tuning_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (directory / "tuning_results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(y_true: list[str], predictions: list[str], directory: Path) -> Path:
    """Save the held-out-test confusion matrix for the selected final configuration."""
    directory.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, predictions, labels=LABELS)
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title("Final tuned Linear SVM - held-out test set")
    figure.tight_layout()
    output_path = directory / "tuning_confusion_matrix_best_model.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1 or not 0 < args.validation_size < 1:
        raise ValueError("--test-size and --validation-size must be between 0 and 1.")
    texts, labels, _, total_cleaned = load_records(args.input, args.max_records, args.random_state)
    x_outer_train, x_test, y_outer_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )
    x_fit, x_validation, y_fit, y_validation = train_test_split(
        x_outer_train,
        y_outer_train,
        test_size=args.validation_size,
        random_state=args.random_state,
        stratify=y_outer_train,
    )

    experiments = experiment_definitions()
    rows: list[dict[str, object]] = []
    for experiment in experiments:
        metrics, _ = evaluate(experiment, x_fit, y_fit, x_validation, y_validation)
        row = {
            "experiment": experiment["id"],
            "model": experiment["model"],
            "ngram_range": str(experiment["ngram_range"]),
            "max_features": experiment["max_features"],
            "class_weight": experiment["class_weight"] or "none",
            "C": experiment["C"],
            **metrics,
        }
        rows.append(row)
        print(f"{row['experiment']}: validation macro_f1={row['macro_f1']:.4f}")

    write_rows(rows, args.metrics_dir)
    best_row = max(rows, key=lambda row: float(row["macro_f1"]))
    best_experiment = next(item for item in experiments if item["id"] == best_row["experiment"])

    # The test set stays untouched until this single final evaluation.
    test_metrics, test_predictions = evaluate(
        best_experiment, x_outer_train, y_outer_train, x_test, y_test
    )
    confusion_matrix_path = save_confusion_matrix(y_test, test_predictions, args.figures_dir)
    final_result = {
        "selected_on_validation": best_row,
        "final_test_metrics": test_metrics,
        "selection_rule": "Highest validation Macro F1",
        "test_set_policy": "The outer test set was not used to choose any configuration.",
        "confusion_matrix_file": str(confusion_matrix_path.resolve()),
        "classification_report": classification_report(y_test, test_predictions, labels=LABELS, zero_division=0),
    }
    (args.metrics_dir / "tuning_final_test_result.json").write_text(
        json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    configuration = {
        "total_cleaned_records": total_cleaned,
        "sample_size": len(texts),
        "random_state": args.random_state,
        "outer_split": {"train_records": len(x_outer_train), "test_records": len(x_test)},
        "inner_split": {"fit_records": len(x_fit), "validation_records": len(x_validation)},
        "selection_metric": "Validation Macro F1",
        "test_set_policy": "Held out until one final evaluation after configuration selection.",
        "experiments": experiments,
    }
    (args.metrics_dir / "tuning_configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Selected: {best_row['experiment']} by validation Macro F1={best_row['macro_f1']:.4f}")
    print(f"Final held-out test Macro F1={test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
