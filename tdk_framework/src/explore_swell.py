import pandas as pd
from pathlib import Path

def explore_file(file_path):
    path = Path(file_path)
    print("=" * 80)
    print("FILE:", path.name)
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".xlsx":
            df = pd.read_excel(path)
        else:
            print("Unsupported file type:", path.suffix)
            return
    except Exception as exc:
        print(f"Error reading {path}: {exc}")
        return

    print("\nColumns:")
    for col in df.columns:
        print(col)
    print("\nFirst 2 rows:")
    print(df.head(2).to_string())
    print("=" * 80)

if __name__ == "__main__":
    base_dir = Path("tdk_framework/data/raw/swell_full")
    files = [
        base_dir / "D - Physiology features (HR_HRV_SCL - final).csv",
        base_dir / "Questionnaire Start Results_final_noBankdata.xlsx",
        base_dir / "Questionnaire Blok Results_final - performance added.xlsx",
    ]

    for file_path in files:
        explore_file(file_path)
