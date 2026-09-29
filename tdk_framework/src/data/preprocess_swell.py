import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SWELL-KW dataset for the Gated Fusion framework"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Path to the raw SWELL-KW data directory",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Path to the directory where the processed dataset will be saved",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        help="Number of rows per sliding window. Default: 2 rows = 2 minutes.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride between consecutive windows in rows. Default: 1 row = 1 minute.",
    )
    args = parser.parse_args()

    if args.raw_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
        raw_dir = base_dir / "data" / "raw" / "swell_full"
    else:
        raw_dir = args.raw_dir

    if args.processed_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
        processed_dir = base_dir / "data" / "processed"
    else:
        processed_dir = args.processed_dir

    processed_dir.mkdir(parents=True, exist_ok=True)

    dynamic_csv = raw_dir / "D - Physiology features (HR_HRV_SCL - final).csv"
    static_xlsx = raw_dir / "Questionnaire Start Results_final_noBankdata.xlsx"

    dyn = pd.read_csv(dynamic_csv)
    stat = pd.read_excel(static_xlsx, engine="openpyxl")

    feature_cols = ["HR", "RMSSD", "SCL"]

    dyn[feature_cols] = dyn[feature_cols].replace([999, 999.0], np.nan)
    dyn = dyn.sort_values(["PP", "timestamp"]).reset_index(drop=True)

    condition_map = {"R": 0, "N": 0, "T": 1, "I": 1}
    dyn["label"] = dyn["Condition"].map(condition_map)

    global_means = dyn[feature_cols].mean()

    for col in feature_cols:
        dyn[col] = dyn.groupby("PP")[col].transform(lambda s: s.ffill().bfill())
        dyn[col] = dyn[col].fillna(global_means[col])

    stat = stat[["PP", "Age", "Gender"]].copy()
    stat["Gender"] = stat["Gender"].map({"m": 0, "f": 1})
    stat["Age"] = stat["Age"].fillna(stat["Age"].median())

    if stat["Gender"].isna().all():
        gender_fill = 0
    else:
        gender_fill = stat["Gender"].mode()[0]
    stat["Gender"] = stat["Gender"].fillna(gender_fill)

    static_features = {}
    for _, row in stat.iterrows():
        subj = row["PP"]
        static_features[subj] = np.array(
            [row["Age"], row["Gender"]], dtype=np.float32
        )

    samples = []
    window_size = args.window_size
    stride = args.stride

    for subj, subj_df in dyn.groupby("PP"):
        subj_df = subj_df.sort_values("timestamp").reset_index(drop=True)

        if subj not in static_features:
            continue

        static_vec = static_features[subj]
        n_rows = len(subj_df)

        if n_rows < window_size:
            continue

        for start in range(0, n_rows - window_size + 1, stride):
            end = start + window_size
            window = subj_df.iloc[start:end]

            label_series = window["label"].dropna()
            if label_series.empty:
                continue

            label = int(label_series.mode().iloc[0])
            if pd.isna(label):
                continue

            features = []

            for col in feature_cols:
                values = window[col].to_numpy(dtype=np.float32)
                if values.size == 0 or np.all(np.isnan(values)):
                    features.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    vals = pd.Series(values).ffill().bfill()
                    mean_val = float(vals.mean())
                    std_val = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                    min_val = float(vals.min())
                    max_val = float(vals.max())
                    features.extend([mean_val, std_val, min_val, max_val])

            hr_vals = window["HR"].to_numpy(dtype=np.float32)
            time_index = np.arange(window_size, dtype=np.float32)
            if np.all(np.isnan(hr_vals)):
                hr_slope = 0.0
            else:
                mask = ~np.isnan(hr_vals)
                if mask.sum() >= 2:
                    hr_slope = float(np.polyfit(time_index[mask], hr_vals[mask], 1)[0])
                else:
                    hr_slope = 0.0
            features.append(hr_slope)

            dynamic_arr = np.array(features, dtype=np.float32).reshape(1, -1)

            samples.append(
                {
                    "subject": subj,
                    "dynamic": dynamic_arr,
                    "static": static_vec,
                    "label": label,
                }
            )

    output_path = processed_dir / "swell_dataset.pt"
    torch.save(samples, output_path)

    labels = [int(s["label"]) for s in samples]
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Saved {len(samples)} samples to {output_path}")
    print("Class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples")
    print(
        "Dynamic feature vector shape per sample:",
        samples[0]["dynamic"].shape if samples else None,
    )


if __name__ == "__main__":
    main()
