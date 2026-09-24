import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description="Preprocess SWELL-KW dataset for the Gated Fusion framework")
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
        default=60,
        help="Number of rows per sliding window",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Stride between consecutive windows",
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

    global_means = dyn[feature_cols].mean()

    condition_map = {"R": 0, "T": 1, "I": 1}
    dyn["label"] = dyn["Condition"].map(condition_map)

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

    for subj in dyn["PP"].unique():
        subj_df = dyn[dyn["PP"] == subj].sort_values("timestamp")
        if subj not in static_features:
            continue

        static_vec = static_features[subj]
        n_rows = len(subj_df)

        for start in range(0, n_rows - window_size + 1, stride):
            end = start + window_size
            window = subj_df.iloc[start:end]
            dynamic_arr = window[feature_cols].to_numpy(dtype=np.float32)

            last_label = window["label"].iloc[-1]
            if pd.isna(last_label):
                continue

            label = int(last_label)

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
    print(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
