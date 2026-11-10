
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
TEST_DIR = DATA_DIR / "test"
LABEL_FILE = DATA_DIR / "labeled_anomalies.csv"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_ID = "P-2"
SPACECRAFT = "SMAP"

# Two annotations found in labeled_anomalies.csv
ANNOTATION_1 = (5350, 6575)
ANNOTATION_2 = (5300, 6420)

# Column 0 is the telemetry signal.
TELEMETRY_FEATURE = 0

# Plot command features separately.
COMMAND_FEATURES = list(range(1, 25))

# Context around the anomaly.
CONTEXT = 300


# ============================================================
# 2. LOAD TELEMETRY AND LABELS
# ============================================================

def load_data():

    telemetry_path = TEST_DIR / f"{CHANNEL_ID}.npy"

    if not telemetry_path.exists():
        raise FileNotFoundError(
            f"Telemetry file not found: {telemetry_path}"
        )

    data = np.load(telemetry_path)

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D telemetry data, got shape {data.shape}"
        )

    labels = pd.read_csv(LABEL_FILE)

    matching_rows = labels[
        (labels["chan_id"] == CHANNEL_ID)
        & (labels["spacecraft"] == SPACECRAFT)
    ]

    if len(matching_rows) != 2:
        raise ValueError(
            f"Expected exactly 2 P-2 annotations, found {len(matching_rows)}"
        )

    print("\nP-2 TELEMETRY INFORMATION")
    print("=" * 55)
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Minimum: {np.min(data):.6f}")
    print(f"Maximum: {np.max(data):.6f}")
    print(f"Mean: {np.mean(data):.6f}")
    print(f"Standard deviation: {np.std(data):.6f}")

    print("\nANNOTATIONS")
    print("=" * 55)

    for i, (_, row) in enumerate(matching_rows.iterrows(), start=1):
        print(f"Annotation {i}")
        print(f"Intervals: {row['anomaly_sequences']}")
        print(f"Class: {row['class']}")

    return data, matching_rows


# ============================================================
# 3. ANNOTATION REGION UTILITIES
# ============================================================

def get_region(data, interval, context=CONTEXT):

    start, end = interval

    left = max(0, start - context)
    right = min(len(data), end + context + 1)

    return left, right


def print_region_statistics(data, interval, name):

    start, end = interval

    # Inclusive endpoint convention used for this inspection.
    anomaly = data[start:end + 1, TELEMETRY_FEATURE]

    print(f"\n{name}")
    print("-" * 55)
    print(f"Start index: {start}")
    print(f"End index: {end}")
    print(f"Number of samples: {len(anomaly)}")
    print(f"Mean: {np.mean(anomaly):.6f}")
    print(f"Median: {np.median(anomaly):.6f}")
    print(f"Minimum: {np.min(anomaly):.6f}")
    print(f"Maximum: {np.max(anomaly):.6f}")
    print(f"Standard deviation: {np.std(anomaly):.6f}")


# ============================================================
# 4. PLOT COMPLETE TELEMETRY WITH BOTH ANNOTATIONS
# ============================================================

def plot_full_telemetry(data):

    signal = data[:, TELEMETRY_FEATURE]
    time = np.arange(len(signal))

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(
        time,
        signal,
        linewidth=0.8,
        label="P-2 telemetry (F0)"
    )

    ax.axvspan(
        ANNOTATION_1[0],
        ANNOTATION_1[1],
        alpha=0.25,
        color="red",
        label="Annotation 1: [5350, 6575]"
    )

    ax.axvspan(
        ANNOTATION_2[0],
        ANNOTATION_2[1],
        alpha=0.25,
        color="orange",
        label="Annotation 2: [5300, 6420]"
    )

    ax.set_title(
        "SMAP P-2: Complete Telemetry with Both Anomaly Annotations"
    )
    ax.set_xlabel("Time index")
    ax.set_ylabel("Telemetry value")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output = OUTPUT_DIR / "p2_full_telemetry_comparison.png"
    fig.savefig(output, dpi=150)
    print(f"\nSaved: {output}")

    plt.close(fig)


# ============================================================
# 5. DETAILED ANOMALY REGION COMPARISON
# ============================================================

def plot_anomaly_region(data):

    signal = data[:, TELEMETRY_FEATURE]
    time = np.arange(len(signal))

    start = min(ANNOTATION_1[0], ANNOTATION_2[0])
    end = max(ANNOTATION_1[1], ANNOTATION_2[1])

    left = max(0, start - CONTEXT)
    right = min(len(signal), end + CONTEXT + 1)

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(
        time[left:right],
        signal[left:right],
        linewidth=1,
        label="P-2 telemetry (F0)"
    )

    ax.axvspan(
        ANNOTATION_1[0],
        ANNOTATION_1[1],
        alpha=0.25,
        color="red",
        label="Annotation 1"
    )

    ax.axvspan(
        ANNOTATION_2[0],
        ANNOTATION_2[1],
        alpha=0.25,
        color="orange",
        label="Annotation 2"
    )

    ax.axvline(
        ANNOTATION_1[0],
        color="red",
        linestyle="--",
        linewidth=1
    )

    ax.axvline(
        ANNOTATION_1[1],
        color="red",
        linestyle="--",
        linewidth=1
    )

    ax.axvline(
        ANNOTATION_2[0],
        color="orange",
        linestyle=":",
        linewidth=1
    )

    ax.axvline(
        ANNOTATION_2[1],
        color="orange",
        linestyle=":",
        linewidth=1
    )

    ax.set_title("P-2 Anomaly Region: Annotation Boundary Comparison")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Telemetry value")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output = OUTPUT_DIR / "p2_anomaly_region_comparison.png"
    fig.savefig(output, dpi=150)
    print(f"\nSaved: {output}")

    plt.close(fig)


# ============================================================
# 6. PLOT COMMAND FEATURES
# ============================================================

def plot_command_features(data):

    features = list(range(1, data.shape[1]))

    print("\nCOMMAND FEATURE INSPECTION")
    print("=" * 60)

    for feature in features:
        values = data[:, feature]
        unique_values = np.unique(values)

        print(
            f"F{feature}: "
            f"unique_values={len(unique_values)}, "
            f"min={values.min():.4f}, "
            f"max={values.max():.4f}"
        )

    fig, axes = plt.subplots(
        4,
        6,
        figsize=(20, 12),
        sharex=True
    )

    axes = axes.flatten()

    time = np.arange(len(data))

    for ax, feature in zip(axes, features):

        ax.step(
            time,
            data[:, feature],
            where="post",
            linewidth=0.6
        )

        ax.axvspan(
            ANNOTATION_1[0],
            ANNOTATION_1[1],
            alpha=0.15,
            color="red"
        )

        ax.axvspan(
            ANNOTATION_2[0],
            ANNOTATION_2[1],
            alpha=0.15,
            color="orange"
        )

        ax.set_title(f"Feature F{feature}")
        ax.grid(alpha=0.2)

    fig.suptitle(
        "SMAP P-2: Command Features with Anomaly Annotations",
        fontsize=15
    )

    fig.supxlabel("Time index")
    fig.supylabel("Command feature value")

    fig.tight_layout()

    output = OUTPUT_DIR / "p2_command_features.png"

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nSaved: {output}")


# ============================================================
# 7. TELEMETRY STATISTICS
# ============================================================

def compare_regions(data):

    signal = data[:, TELEMETRY_FEATURE]

    regions = {
        "Before annotations": (4800, 5200),
        "Annotation 1 only": (6421, 6575),
        "Annotation 2 only": (5300, 5349),
        "Common overlap": (5350, 6420),
        "After annotations": (6700, 7100),
    }

    print("\nREGION-WISE TELEMETRY STATISTICS")
    print("=" * 75)

    for name, (start, end) in regions.items():

        values = signal[start:end + 1]

        print(f"\n{name}: [{start}, {end}]")
        print(f"Samples: {len(values)}")
        print(f"Mean: {np.mean(values):.6f}")
        print(f"Std: {np.std(values):.6f}")
        print(f"Min: {np.min(values):.6f}")
        print(f"Max: {np.max(values):.6f}")


# ============================================================
# 8. MAIN
# ============================================================

def main():

    data, matching_rows = load_data()

    # Verify that the manually configured intervals match CSV rows.
    csv_intervals = []

    import ast

    for _, row in matching_rows.iterrows():
        csv_intervals.extend(
            ast.literal_eval(row["anomaly_sequences"])
        )

    csv_intervals = [tuple(interval) for interval in csv_intervals]

    configured_intervals = [
        ANNOTATION_1,
        ANNOTATION_2
    ]

    if sorted(csv_intervals) != sorted(configured_intervals):
        raise ValueError(
            "Configured intervals do not match CSV annotations."
        )

    # Ensure intervals fit inside telemetry sequence.
    for start, end in configured_intervals:
        if not (0 <= start <= end < len(data)):
            raise ValueError(
                f"Invalid interval {(start, end)} for data length {len(data)}"
            )

    print_region_statistics(
        data,
        ANNOTATION_1,
        "ANNOTATION 1"
    )

    print_region_statistics(
        data,
        ANNOTATION_2,
        "ANNOTATION 2"
    )

    compare_regions(data)

    plot_full_telemetry(data)

    plot_anomaly_region(data)

    plot_command_features(data)

    print("\nP-2 visualization completed.")
    print(f"Plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
