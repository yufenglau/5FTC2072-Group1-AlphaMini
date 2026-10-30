"""ONE-TIME MIGRATION - run this once after updating common.py's extract_features.

Adds the 14 new distance/curl columns to your existing data/landmarks.csv by
computing them from the 63 normalized coordinates you already collected -
no need to redo data collection with your webcam.

Run:  python migrate_add_features.py
Then: python train_model.py   (as usual)
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "landmarks.csv")

FINGERTIPS = [4, 8, 12, 16, 20]
MCPS = [2, 5, 9, 13, 17]


def extra_features_from_row(row):
    # Rebuild the (21, 3) coords array from the flat x0,y0,z0,...,x20,y20,z20 columns.
    coords = np.array(
        [[row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]] for i in range(21)],
        dtype=np.float32,
    )

    feats = []
    for a, b in zip(FINGERTIPS, FINGERTIPS[1:]):
        feats.append(float(np.linalg.norm(coords[a] - coords[b])))
    for tip in FINGERTIPS:
        feats.append(float(np.linalg.norm(coords[tip])))
    for tip, mcp in zip(FINGERTIPS, MCPS):
        feats.append(float(np.linalg.norm(coords[tip] - coords[mcp])))
    return feats


def main():
    if not os.path.exists(DATA_PATH):
        print(f"No file at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    new_cols = (
        [f"fintip_dist_{i}" for i in range(4)]
        + [f"tip_wrist_dist_{i}" for i in range(5)]
        + [f"curl_{i}" for i in range(5)]
    )

    if new_cols[0] in df.columns:
        print("Already migrated (new columns already present). Nothing to do.")
        return

    print(f"Computing {len(new_cols)} new columns for {len(df)} existing samples...")
    extra_rows = df.apply(extra_features_from_row, axis=1, result_type="expand")
    extra_rows.columns = new_cols

    out = pd.concat([df, extra_rows], axis=1)
    out.to_csv(DATA_PATH, index=False)
    print(f"Done. {DATA_PATH} now has {out.shape[1] - 1} feature columns (was {df.shape[1] - 1}).")
    print("Next: python train_model.py")


if __name__ == "__main__":
    main()
