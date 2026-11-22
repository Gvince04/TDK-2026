import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TDK_FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = TDK_FRAMEWORK_ROOT / "data" / "processed"
DEFAULT_RESULTS_DIR = TDK_FRAMEWORK_ROOT / "results"
DEFAULT_WESAD_RAW_DIR = TDK_FRAMEWORK_ROOT / "data" / "raw" / "wesad"

if str(TDK_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(TDK_FRAMEWORK_ROOT))

GENDER_LABELS = {0: "Male", 1: "Female"}
WESAD_GENDER_MAP = {"male": 0, "m": 0, "female": 1, "f": 1}


def _load_pt(path: Path):
    return torch.load(path, weights_only=False)


def _extract_subject_static(dataset) -> Tuple[List, np.ndarray]:
    if hasattr(dataset, "subject_id") and hasattr(dataset, "X_stat"):
        subjects = list(dataset.subject_id)
        static = dataset.X_stat.numpy()
    elif isinstance(dataset, list):
        subjects = [s["subject"] for s in dataset]
        static = np.stack([np.asarray(s["static"]).reshape(-1) for s in dataset])
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    df = pd.DataFrame({"subject_id": subjects})
    for i in range(static.shape[1]):
        df[f"static_{i}"] = static[:, i]

    dedup = df.drop_duplicates(subset=["subject_id"]).reset_index(drop=True)
    return dedup["subject_id"].tolist(), dedup.drop(columns=["subject_id"]).to_numpy()


def _infer_schema(dataset_name: str, static: np.ndarray) -> Optional[Dict[str, int]]:
    name = dataset_name.lower()
    if "swell" in name and static.shape[1] >= 2:
        return {"age": 0, "gender": 1}
    return None


def _parse_wesad_readme(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    age_match = re.search(r"Age:\s*(\d+)", text)
    gender_match = re.search(r"Gender:\s*(\w+)", text, re.IGNORECASE)
    if not age_match or not gender_match:
        return None
    gender_raw = gender_match.group(1).strip().lower()
    gender_code = WESAD_GENDER_MAP.get(gender_raw)
    if gender_code is None:
        return None
    return {"age": int(age_match.group(1)), "gender": gender_code}


def load_wesad_demographics(raw_dir: Path) -> pd.DataFrame:
    rows = []
    if not raw_dir.exists():
        return pd.DataFrame(columns=["subject_id", "age", "gender"])
    for subject_dir in sorted(raw_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("S"):
            continue
        readme = subject_dir / f"{subject_dir.name}_readme.txt"
        parsed = _parse_wesad_readme(readme)
        if parsed is None:
            continue
        rows.append(
            {
                "subject_id": subject_dir.name,
                "age": parsed["age"],
                "gender": parsed["gender"],
            }
        )
    return pd.DataFrame(rows)


def _static_varies_within_subjects(dataset) -> bool:
    if not (hasattr(dataset, "subject_id") and hasattr(dataset, "X_stat")):
        return False
    subjects = np.asarray(dataset.subject_id)
    static = dataset.X_stat.numpy()
    if static.ndim == 1:
        static = static.reshape(-1, 1)
    for subj in np.unique(subjects):
        mask = subjects == subj
        if np.unique(static[mask], axis=0).shape[0] > 1:
            return True
    return False


def summarize_varying_static(dataset, dataset_name: str) -> str:
    subjects = np.asarray(dataset.subject_id)
    static = dataset.X_stat.numpy()
    if static.ndim == 1:
        static = static.reshape(-1, 1)

    n_subjects = len(np.unique(subjects))
    n_windows = len(subjects)
    lines = [f"=== {dataset_name} (N={n_subjects}, ablakok={n_windows}) ==="]
    lines.append("Statikus jellemzo: feladatonkent valtozo kontextus (nem demografiai)")

    if static.shape[1] == 1:
        values = static[:, 0]
        unique_vals, counts = np.unique(values, return_counts=True)
        lines.append("Szintek eloszlasa:")
        for val, count in zip(unique_vals, counts):
            pct = 100.0 * count / n_windows
            lines.append(f"  {val}: {count} ablak ({pct:.1f}%)")

        per_subject_levels = []
        for subj in np.unique(subjects):
            per_subject_levels.append(len(np.unique(values[subjects == subj])))
        per_subject_levels = np.asarray(per_subject_levels)
        all_levels = len(unique_vals)
        full_coverage = int((per_subject_levels == all_levels).sum())
        lines.append(
            f"Resztvevok, akik minden szintet megeltek: {full_coverage} / {n_subjects}"
        )
        lines.append(
            f"Egy resztvevore juto egyedi szintek: "
            f"atlag {per_subject_levels.mean():.2f}, "
            f"min {per_subject_levels.min()}, max {per_subject_levels.max()}"
        )
    else:
        unique_vectors = np.unique(static, axis=0).shape[0]
        lines.append(f"Egyedi statikus vektorok szama: {unique_vectors}")

    return "\n".join(lines)


def summarize_static(df: pd.DataFrame, dataset_name: str) -> str:
    n = len(df)
    lines = [f"=== {dataset_name} (N={n}) ==="]

    if "age" in df.columns and df["age"].notna().any():
        age = df["age"].dropna()
        lines.append(
            f"Életkor: {age.mean():.2f} ± {age.std():.2f} év | "
            f"Tartomány: [{age.min():.0f}, {age.max():.0f}]"
        )
        lines.append(
            f"Medián: {age.median():.1f} | "
            f"IQR: [{age.quantile(0.25):.1f}, {age.quantile(0.75):.1f}]"
        )
    else:
        lines.append("Életkor: nem elérhető")

    if "gender" in df.columns and df["gender"].notna().any():
        counts = df["gender"].dropna().astype(int).map(
            lambda g: GENDER_LABELS.get(g, str(g))
        ).value_counts()
        lines.append("Nemek:")
        for label, count in counts.items():
            lines.append(f"  {label}: {count}")
    else:
        lines.append("Nemek: nem elérhető")

    static_cols = [c for c in df.columns if c.startswith("static_")]
    if "age" in df.columns and "gender" in df.columns:
        unique_states = df[["age", "gender"]].drop_duplicates().shape[0]
        lines.append(
            f"Egyedi (életkor, nem) kombinációk száma: {unique_states} / {n}"
        )
    if static_cols:
        unique_vectors = df[static_cols].drop_duplicates().shape[0]
        lines.append(f"Egyedi statikus vektorok száma: {unique_vectors} / {n}")

    return "\n".join(lines)


def build_dataframe(dataset_name: str, dataset) -> pd.DataFrame:
    subjects, static = _extract_subject_static(dataset)
    df = pd.DataFrame({"subject_id": subjects})
    for i in range(static.shape[1]):
        df[f"static_{i}"] = static[:, i]

    schema = _infer_schema(dataset_name, static)
    if schema is not None:
        df["age"] = static[:, schema["age"]]
        df["gender"] = static[:, schema["gender"]]

    return df


def discover_datasets(processed_dir: Path) -> Dict[str, Path]:
    if not processed_dir.exists():
        return {}
    return {p.stem: p for p in sorted(processed_dir.glob("*.pt"))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize subject-level static features across processed datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names (stem of .pt file) or explicit paths. Default: all in processed dir.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing processed .pt datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "static_summary.txt",
        help="Path to save the summary text.",
    )
    args = parser.parse_args()

    available = discover_datasets(args.processed_dir)
    if not available:
        raise FileNotFoundError(f"No .pt datasets found in {args.processed_dir}")

    if args.datasets:
        targets = {}
        for name in args.datasets:
            p = Path(name)
            if p.exists():
                targets[p.stem] = p
            elif name in available:
                targets[name] = available[name]
            else:
                raise FileNotFoundError(f"Dataset not found: {name}")
    else:
        targets = available

    blocks = []
    for name, path in targets.items():
        dataset = _load_pt(path)
        if _static_varies_within_subjects(dataset):
            blocks.append(summarize_varying_static(dataset, name))
            continue
        df = build_dataframe(name, dataset)
        blocks.append(summarize_static(df, name))

    wesad_df = load_wesad_demographics(DEFAULT_WESAD_RAW_DIR)
    if not wesad_df.empty:
        wesad_block = summarize_static(wesad_df, "WESAD (raw readme)")
        wesad_block += (
            "\nMegjegyzes: a .pt fajl statikus vektora 5 dimenzios szintetikus "
            "random ertek; a demografia a nyers readme fajlokbol szarmazik."
        )
        blocks.append(wesad_block)

    output_text = "\n\n".join(blocks)
    print(output_text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_text + "\n", encoding="utf-8")
    print(f"\nSaved summary to: {args.output}")


if __name__ == "__main__":
    main()
