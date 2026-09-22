import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "tdk_framework" / "results"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "tdk_framework" / "plots"

PARADIGMS = ["zero_shot", "intra_subject", "few_shot"]
PARADIGM_LABELS = {
    "zero_shot": "Zero-shot",
    "intra_subject": "Intra-subject",
    "few_shot": "Few-shot",
}
METRICS = {
    "balanced_acc": "Balanced Accuracy",
    "f1": "Macro F1",
    "auroc": "AUROC",
}
MODEL_ORDER = ["baseline", "gated_fusion"]


def _parse_result_filename(path: Path):
    stem = path.stem
    prefix = "experiment_results_"
    if not stem.startswith(prefix):
        return None

    remainder = stem[len(prefix):]
    for model in MODEL_ORDER:
        suffix = f"_{model}"
        if remainder.endswith(suffix):
            dataset = remainder[: -len(suffix)]
            if dataset:
                return dataset, model

    return None


def load_results(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []

    for path in sorted(results_dir.glob("experiment_results_*.json")):
        parsed = _parse_result_filename(path)
        if parsed is None:
            continue

        dataset, model = parsed
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for paradigm in PARADIGMS:
            paradigm_result = payload.get(paradigm)
            if not paradigm_result:
                continue

            aggregates = paradigm_result.get("aggregates", {})
            for metric_key, metric_label in METRICS.items():
                mean = aggregates.get(f"{metric_key}_mean")
                std = aggregates.get(f"{metric_key}_std")
                if mean is None:
                    continue

                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "paradigm": paradigm,
                    "metric": metric_label,
                    "mean": float(mean),
                    "std": float(std) if std is not None else 0.0,
                })

    return pd.DataFrame(rows)


def plot_metric_comparison(
    df: pd.DataFrame,
    dataset: str,
    metric_label: str,
    output_path: Path,
) -> None:
    subset = df[(df["dataset"] == dataset) & (df["metric"] == metric_label)]
    if subset.empty:
        return

    subset = subset.copy()
    subset["paradigm_label"] = subset["paradigm"].map(PARADIGM_LABELS)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sns.barplot(
        data=subset,
        x="paradigm_label",
        y="mean",
        hue="model",
        order=[PARADIGM_LABELS[p] for p in PARADIGMS],
        hue_order=MODEL_ORDER,
        errorbar=None,
        ax=ax,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

    ax.set_xlabel("")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{dataset.upper()} - {metric_label} by paradigm")
    ax.legend(title="Model", loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot experiment results as publication-ready bar charts."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing experiment_results_*.json files.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory where the generated plots will be saved.",
    )
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        raise ValueError(f"No experiment results found in {args.results_dir}")

    args.plots_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(df["dataset"].unique()):
        for metric_label in METRICS.values():
            metric_slug = metric_label.lower().replace(" ", "_")
            output_path = args.plots_dir / f"{dataset}_{metric_slug}_comparison.png"
            plot_metric_comparison(df, dataset, metric_label, output_path)
            print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
