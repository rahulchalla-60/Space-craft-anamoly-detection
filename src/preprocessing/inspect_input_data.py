from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
LABELS_FILE = DATA_DIR / "labeled_anomalies.csv"


def inspect_arrays(directory, name):
    print(f"\n{'=' * 70}")
    print(f"{name} DATA INSPECTION")
    print(f"{'=' * 70}")

    files = sorted(directory.glob("*.npy"))

    print(f"Total channels: {len(files)}")

    if not files:
        print(f"No .npy files found in {directory}")
        return

    for file in files:
        data = np.load(file)

        print(f"\nChannel: {file.stem}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Minimum: {np.nanmin(data):.4f}")
        print(f"Maximum: {np.nanmax(data):.4f}")
        print(f"NaN values: {np.isnan(data).sum()}")
        print(f"Infinite values: {np.isinf(data).sum()}")

        if data.ndim == 2:
            print(f"Time steps: {data.shape[0]}")
            print(f"Features: {data.shape[1]}")


def inspect_labels():
    print(f"\n{'=' * 70}")
    print("ANOMALY LABEL INSPECTION")
    print(f"{'=' * 70}")

    if not LABELS_FILE.exists():
        print("labeled_anomalies.csv not found!")
        return

    df = pd.read_csv(LABELS_FILE)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst 10 records:")
    print(df.head(10).to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum())


def main():
    inspect_arrays(TRAIN_DIR, "TRAINING")
    inspect_arrays(TEST_DIR, "TESTING")
    inspect_labels()


if __name__ == "__main__":
    main()

