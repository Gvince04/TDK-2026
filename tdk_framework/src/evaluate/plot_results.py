import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(results_file: Path) -> dict:
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    results_dir = Path(__file__).resolve().parents[2] / "results"
    results_file = results_dir / "experiment_results.json"

    try:
        results = load_results(results_file)
    except FileNotFoundError:
        print("Missing results file. Run run_all.py first.")
        return

    sns.set_theme(style="whitegrid")

    bar_rows = []
    for paradigm, result in results.items():
        if not isinstance(result, dict):
            continue
        aggregates = result.get("aggregates", {})
        auroc_mean = aggregates.get("auroc_mean")
        f1_mean = aggregates.get("f1_mean")

        if auroc_mean is not None:
            bar_rows.append({
                "Paradigm": paradigm,
                "Metric": "AUROC",
                "Value": auroc_mean,
            })
        if f1_mean is not None:
            bar_rows.append({
                "Paradigm": paradigm,
                "Metric": "Macro-F1",
                "Value": f1_mean,
            })

    if bar_rows:
        df_bar = pd.DataFrame(bar_rows)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_bar, x="Paradigm", y="Value", hue="Metric")
        plt.title("Average AUROC and Macro-F1 across paradigms")
        plt.ylabel("Score")
        plt.tight_layout()
        results_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(results_dir / "average_metrics.png", dpi=300)
        plt.close()
    else:
        print("No aggregate metrics available for bar chart.")

    box_rows = []
    for paradigm, result in results.items():
        if not isinstance(result, dict):
            continue
        folds = result.get("folds", [])
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            auroc = fold.get("auroc")
            if auroc is not None:
                box_rows.append({
                    "Paradigm": paradigm,
                    "AUROC": auroc,
                })

    if box_rows:
        df_box = pd.DataFrame(box_rows)
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df_box, x="Paradigm", y="AUROC")
        plt.title("Per-subject AUROC distribution across paradigms")
        plt.ylabel("AUROC")
        plt.tight_layout()
        results_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(results_dir / "per_subject_auroc.png", dpi=300)
        plt.close()
    else:
        print("No per-subject AUROC values available for boxplot.")


if __name__ == "__main__":
    main()
