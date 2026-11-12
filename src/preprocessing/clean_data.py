from pathlib import Path
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "processed" / "clean"
TRAIN_OUTPUT = OUTPUT_DIR / "train"
TEST_OUTPUT = OUTPUT_DIR / "test"


def get_train_medians(data):
    """Calculate median of each feature using finite training values."""
    medians = np.zeros(data.shape[1], dtype=np.float64)
    for i in range(data.shape[1]):
        valid_values = data[np.isfinite(data[:, i]), i]
        if len(valid_values) > 0:
            medians[i] = np.median(valid_values)
        else:
            medians[i] = 0.0
    return medians


def clean_array(data, medians):
    """Replace NaN and infinite values with training medians."""
    data = data.astype(np.float64, copy=True)
    for i in range(data.shape[1]):
        invalid_mask = ~np.isfinite(data[:, i])
        if np.any(invalid_mask):
            data[invalid_mask, i] = medians[i]
    return data.astype(np.float32)


def find_constant_features(data):
    """Identify features with zero variation (constant values)."""
    constant_features = []
    for i in range(data.shape[1]):
        col = data[:, i]
        if np.all(col == col[0]):
            constant_features.append(i)
    return constant_features


def find_duplicate_features(data):
    """Identify pairs of features that have identical values across all time steps."""
    duplicate_pairs = []
    n_features = data.shape[1]
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if np.array_equal(data[:, i], data[:, j]):
                duplicate_pairs.append((i, j))
    return duplicate_pairs


def main():
    print("=" * 70)
    print("PHASE 2.2 — TELEMETRY DATA CLEANING")
    print("=" * 70)

    # Ensure output directories exist
    TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)

    train_files = sorted(TRAIN_DIR.glob("*.npy"))
    test_files = sorted(TEST_DIR.glob("*.npy"))

    train_channels = {f.stem: f for f in train_files}
    test_channels = {f.stem: f for f in test_files}

    common_channels = sorted(set(train_channels.keys()) & set(test_channels.keys()))
    print(f"\nDiscovered {len(train_files)} train files and {len(test_files)} test files.")
    print(f"Processing {len(common_channels)} matching channels...")

    total_train_invalids = 0
    total_test_invalids = 0
    channels_with_constants = {}
    channels_with_duplicates = {}

    for channel in common_channels:
        train_raw = np.load(train_channels[channel])
        test_raw = np.load(test_channels[channel])

        # Track invalid values count before cleaning
        train_invalids = int(np.sum(~np.isfinite(train_raw)))
        test_invalids = int(np.sum(~np.isfinite(test_raw)))
        total_train_invalids += train_invalids
        total_test_invalids += test_invalids

        # Compute training medians
        medians = get_train_medians(train_raw)

        # Clean both arrays with training medians and cast to float32
        train_clean = clean_array(train_raw, medians)
        test_clean = clean_array(test_raw, medians)

        # Sanity checks
        assert train_clean.shape == train_raw.shape, f"Train shape changed for {channel}"
        assert test_clean.shape == test_raw.shape, f"Test shape changed for {channel}"
        assert train_clean.dtype == np.float32, f"Train dtype not float32 for {channel}"
        assert test_clean.dtype == np.float32, f"Test dtype not float32 for {channel}"
        assert np.all(np.isfinite(train_clean)), f"Non-finite values remain in clean train for {channel}"
        assert np.all(np.isfinite(test_clean)), f"Non-finite values remain in clean test for {channel}"

        # Check for constant and duplicate features (recorded from training data)
        const_feats = find_constant_features(train_clean)
        if const_feats:
            channels_with_constants[channel] = const_feats

        dup_feats = find_duplicate_features(train_clean)
        if dup_feats:
            channels_with_duplicates[channel] = dup_feats

        # Save cleaned arrays
        np.save(TRAIN_OUTPUT / f"{channel}.npy", train_clean)
        np.save(TEST_OUTPUT / f"{channel}.npy", test_clean)

    print("\nCLEANING SUMMARY:")
    print("-" * 50)
    print(f"Total channels processed: {len(common_channels)}")
    print(f"Total invalid values replaced (Train): {total_train_invalids}")
    print(f"Total invalid values replaced (Test):  {total_test_invalids}")
    print(f"Clean arrays dtype: {np.float32}")
    print(f"Channels with constant features: {len(channels_with_constants)}")
    print(f"Channels with duplicate features: {len(channels_with_duplicates)}")

    if channels_with_constants:
        print("\nChannels with constant features (Feature indices):")
        for ch, feats in list(channels_with_constants.items())[:10]:
            print(f"  {ch}: {feats}")
        if len(channels_with_constants) > 10:
            print(f"  ... and {len(channels_with_constants) - 10} more channels.")

    if channels_with_duplicates:
        print("\nChannels with duplicate feature pairs (i, j):")
        for ch, pairs in list(channels_with_duplicates.items())[:10]:
            print(f"  {ch}: {pairs}")
        if len(channels_with_duplicates) > 10:
            print(f"  ... and {len(channels_with_duplicates) - 10} more channels.")

    print(f"\nAll cleaned arrays successfully saved to:")
    print(f"  Train: {TRAIN_OUTPUT}")
    print(f"  Test:  {TEST_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()

