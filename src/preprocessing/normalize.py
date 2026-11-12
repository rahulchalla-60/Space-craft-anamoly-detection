from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "artifacts" / "processed" / "clean"
TRAIN_DIR = INPUT_DIR / "train"
TEST_DIR = INPUT_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "processed" / "normalized"
TRAIN_OUTPUT = OUTPUT_DIR / "train"
TEST_OUTPUT = OUTPUT_DIR / "test"
SCALER_OUTPUT = OUTPUT_DIR / "scalers"


def normalize_channel(train_file, test_file):
    channel = train_file.stem
    print(f"\nNormalizing channel: {channel}")

    train_data = np.load(train_file)
    test_data = np.load(test_file)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    train_scaled = train_scaled.astype(np.float32)
    test_scaled = test_scaled.astype(np.float32)

    # Save normalized arrays
    np.save(TRAIN_OUTPUT / train_file.name, train_scaled)
    np.save(TEST_OUTPUT / test_file.name, test_scaled)

    # Save fitted scaler
    joblib.dump(scaler, SCALER_OUTPUT / f"{channel}_scaler.pkl")

    # Display statistics
    print(f"Original shape: {train_data.shape}")
    print(f"Normalized shape: {train_scaled.shape}")
    print(f"Training mean: {np.mean(train_scaled, axis=0)[:5]}")
    print(f"Training std: {np.std(train_scaled, axis=0)[:5]}")

    assert np.isfinite(train_scaled).all()
    assert np.isfinite(test_scaled).all()


def main():
    TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    SCALER_OUTPUT.mkdir(parents=True, exist_ok=True)

    train_files = sorted(TRAIN_DIR.glob("*.npy"))
    print("Starting normalization...\n")

    processed = 0
    for train_file in train_files:
        test_file = TEST_DIR / train_file.name
        if not test_file.exists():
            print(f"Skipping {train_file.stem}: test file missing")
            continue

        normalize_channel(train_file, test_file)
        processed += 1

    print("\n" + "=" * 50)
    print("NORMALIZATION COMPLETED")
    print("=" * 50)
    print(f"Processed channels: {processed}")
    print(f"Normalized data: {OUTPUT_DIR}")
    print(f"Saved scalers: {SCALER_OUTPUT}")


if __name__ == "__main__":
    main()

