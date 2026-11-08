
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN = ROOT / "data" / "train"
TEST = ROOT / "data" / "test"

records = []

channels = sorted(
    {p.stem for p in TRAIN.glob("*.npy")} |
    {p.stem for p in TEST.glob("*.npy")}
)

for channel in channels:

    row: dict[str, object] = {"channel": channel}

    for split, folder in [
        ("train", TRAIN),
        ("test", TEST)
    ]:

        path = folder / f"{channel}.npy"

        if not path.exists():
            row[f"{split}_exists"] = False
            continue

        try:
            data = np.load(path, allow_pickle=False)

            finite = data[np.isfinite(data)]

            row[f"{split}_exists"] = True
            row[f"{split}_shape"] = str(data.shape)
            row[f"{split}_dtype"] = str(data.dtype)
            row[f"{split}_observations"] = data.shape[0]
            row[f"{split}_features"] = (
                data.shape[1] if data.ndim > 1 else 1
            )

            row[f"{split}_nan"] = int(np.isnan(data).sum())
            row[f"{split}_inf"] = int(np.isinf(data).sum())

            if finite.size:
                row[f"{split}_mean"] = float(finite.mean())
                row[f"{split}_std"] = float(finite.std())
                row[f"{split}_min"] = float(finite.min())
                row[f"{split}_max"] = float(finite.max())

        except Exception as error:
            row[f"{split}_error"] = str(error)

    records.append(row)

df = pd.DataFrame(records)

output_dir = ROOT / "artifacts" / "reports"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "channel_profile.csv"

df.to_csv(output_path, index=False)

print("=" * 70)
print("ALL CHANNELS PROFILE")
print("=" * 70)

print("\nTotal channels:", len(df))

print("\nShape distribution:")

for column in ["train_shape", "test_shape"]:
    if column in df.columns:
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False))

print("\nMissing training files:",
      (~df["train_exists"].fillna(False)).sum())

print("Missing testing files:",
      (~df["test_exists"].fillna(False)).sum())

print("\nChannels with invalid values:")

for split in ["train", "test"]:
    nan_col = f"{split}_nan"
    inf_col = f"{split}_inf"

    if nan_col in df.columns:
        invalid = df[
            (df[nan_col] > 0) | (df[inf_col] > 0)
        ]

        print(f"{split}: {len(invalid)} channels")

print("\nSaved:", output_path)
print("\nProfile completed.")
