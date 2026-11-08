
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN = ROOT / "data" / "train"
TEST = ROOT / "data" / "test"


def inspect_file(path):
    print("\n" + "=" * 65)
    print("FILE:", path.name)
    print("=" * 65)

    data = np.load(path, allow_pickle=False)

    print("Shape:", data.shape)
    print("Dimensions:", data.ndim)
    print("Data type:", data.dtype)
    print("Number of elements:", data.size)

    print("\nFirst 5 observations:")
    print(data[:5])

    print("\nLast 5 observations:")
    print(data[-5:])

    print("\nData quality:")
    print("NaN count:", np.isnan(data).sum())
    print("Positive infinity:",
          np.isposinf(data).sum())
    print("Negative infinity:",
          np.isneginf(data).sum())

    finite = data[np.isfinite(data)]

    if finite.size:
        print("\nStatistical information:")
        print("Minimum:", finite.min())
        print("Maximum:", finite.max())
        print("Mean:", finite.mean())
        print("Median:", np.median(finite))
        print("Standard deviation:", finite.std())

    if data.ndim == 2:
        print("\nFEATURE-LEVEL STATISTICS")

        for i in range(data.shape[1]):
            column = data[:, i]
            values = column[np.isfinite(column)]

            if values.size == 0:
                print(f"Feature {i}: No finite values")
                continue

            print(
                f"Feature {i}: "
                f"mean={values.mean():.4f}, "
                f"std={values.std():.4f}, "
                f"min={values.min():.4f}, "
                f"max={values.max():.4f}, "
                f"unique={np.unique(values).size}"
            )


# Inspect representative channels
train_files = sorted(TRAIN.glob("*.npy"))
test_files = sorted(TEST.glob("*.npy"))

if not train_files or not test_files:
    raise FileNotFoundError("Training or testing files missing.")

# Choose a channel available in both directories
channel = train_files[0].stem

train_file = TRAIN / f"{channel}.npy"
test_file = TEST / f"{channel}.npy"

inspect_file(train_file)
inspect_file(test_file)

print("\nSelected channel:", channel)
