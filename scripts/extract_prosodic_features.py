"""
Standalone script to extract prosodic features and cache them to disk.

Run from the project root:
    python scripts/extract_prosodic_features.py <split_csv> <cache_dir>

<split_csv>  : CSV file with at least a FULL_FILE_PATH column (one audio file per row).
<cache_dir>  : Directory where per-file .pkl caches are stored.

The script writes one .pkl file per audio file (skipping files that are already cached),
then writes a combined <split_csv>.feats.csv with all features + metadata columns.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the path so modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import joblib
from tqdm import tqdm
from modules.audio_processing import extract_prosodic_features


def get_features(file_path: str, cache_dir: str) -> dict:
    safe_name = file_path.replace("/", "_").replace(":", "_")
    cache_path = os.path.join(cache_dir, safe_name + ".pkl")

    if os.path.exists(cache_path):
        return joblib.load(cache_path)

    feats = extract_prosodic_features(file_path)
    joblib.dump(feats, cache_path)
    return feats


def main():
    split_csv = sys.argv[1]
    cache_dir = sys.argv[2]
    os.makedirs(cache_dir, exist_ok=True)

    df = pd.read_csv(split_csv)
    records = []
    for _, row in tqdm(df.iterrows(), desc="Extracting features", total=len(df)):
        feats = get_features(row["FULL_FILE_PATH"], cache_dir)
        feats["ATTACK_LABEL"] = row["ATTACK_LABEL"]
        feats["LABEL"] = row["LABEL"]
        records.append(feats)

    out_path = split_csv.replace(".csv", ".feats.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Saved {len(records)} rows to {out_path}")


if __name__ == "__main__":
    main()
