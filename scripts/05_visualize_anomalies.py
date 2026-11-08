
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
TRAIN = DATA / "train"
TEST = DATA / "test"

OUTPUT = ROOT / "artifacts" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Change these after inspecting the label CSV
CHANNEL = "A-1"
FEATURE = 0

# Load telemetry
train = np.load(
    TRAIN / f"{CHANNEL}.npy",
    allow_pickle=False
)

test = np.load(
    TEST / f"{CHANNEL}.npy",
    allow_pickle=False
)

if train.ndim == 1:
    train = train.reshape(-1, 1)

if test.ndim == 1:
    test = test.reshape(-1, 1)

if FEATURE >= test.shape[1]:
    raise ValueError("Selected feature does not exist.")

# Load labels
labels = pd.read_csv(DATA / "labeled_anomalies.csv")

row = labels[
    labels["chan_id"].astype(str) == CHANNEL
]

if row.empty:
    raise ValueError(f"No anomaly labels for {CHANNEL}")

label = row.iloc[0]

intervals = label["anomaly_sequences"]

if isinstance(intervals, str):
    intervals = ast.literal_eval(intervals)

# Plot training signal
plt.figure(figsize=(15, 5))

plt.plot(
    train[:, FEATURE],
    linewidth=0.6,
    label="Training telemetry"
)

plt.title(f"{CHANNEL} - Training Telemetry")
plt.xlabel("Time-step index")
plt.ylabel(f"Feature {FEATURE}")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT / f"{CHANNEL}_training.png",
    dpi=150
)

plt.close()

# Plot testing signal with anomaly labels
plt.figure(figsize=(15, 5))

plt.plot(
    test[:, FEATURE],
    linewidth=0.7,
    label="Testing telemetry"
)

for i, interval in enumerate(intervals):
    start, end = map(int, interval)

    plt.axvspan(
        start,
        end,
        color="red",
        alpha=0.2,
        label="Labeled anomaly" if i == 0 else None
    )

plt.title(
    f"{CHANNEL} - Testing Telemetry "
    f"({label['class']} anomaly)"
)

plt.xlabel("Time-step index")
plt.ylabel(f"Feature {FEATURE}")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT / f"{CHANNEL}_testing_anomalies.png",
    dpi=150
)

plt.show()

plt.close()

# Compare distributions
plt.figure(figsize=(10, 5))

plt.hist(
    train[:, FEATURE],
    bins=50,
    alpha=0.6,
    density=True,
    label="Training"
)

plt.hist(
    test[:, FEATURE],
    bins=50,
    alpha=0.6,
    density=True,
    label="Testing"
)

plt.title(f"{CHANNEL} - Train vs Test Distribution")
plt.xlabel("Telemetry value")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT / f"{CHANNEL}_distribution.png",
    dpi=150
)

plt.show()
plt.close()

print("Channel:", CHANNEL)
print("Spacecraft:", label["spacecraft"])
print("Anomaly class:", label["class"])
print("Anomaly intervals:", intervals)
print("Training shape:", train.shape)
print("Testing shape:", test.shape)

print("\nVisualizations saved in:", OUTPUT)
