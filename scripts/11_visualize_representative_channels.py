
from pathlib import Path
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

LABEL_FILE = DATA_DIR / "labeled_anomalies.csv"

PLOT_DIR = ROOT / "artifacts" / "plots"
REPORT_DIR = ROOT / "artifacts" / "reports"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Representative channels
SELECTED_CHANNELS = [
    "P-2",
    "S-1",
    "E-2",
    "M-1",
]

# We initially focus on F0 (telemetry).
TELEMETRY_FEATURE = 0


# ============================================================
# 2. LOAD LABELS
# ============================================================

def load_labels():

    if not LABEL_FILE.exists():
        raise FileNotFoundError(LABEL_FILE)

    df = pd.read_csv(LABEL_FILE)

    print("\nLABEL DATASET")
    print("=" * 65)
    print(f"Total records: {len(df)}")

    return df


def get_channel_labels(labels, channel_id):

    matching = labels[
        labels["chan_id"] == channel_id
    ]

    if matching.empty:
        return []

    intervals = []

    for _, row in matching.iterrows():

        parsed = ast.literal_eval(
            row["anomaly_sequences"]
        )

        classes = [
            item.strip().strip("'\"")
            for item in str(row["class"]).strip("[]").split(",")
        ]

        for interval, anomaly_class in zip(parsed, classes):

            start, end = map(int, interval)

            intervals.append({
                "start": start,
                "end": end,
                "class": anomaly_class,
                "spacecraft": row["spacecraft"],
            })

    return intervals


# ============================================================
# 3. LOAD CHANNEL ARRAYS
# ============================================================

def load_channel(channel_id, folder):

    path = folder / f"{channel_id}.npy"

    if not path.exists():
        return None

    return np.load(path)


# ============================================================
# 4. CHANNEL INFORMATION
# ============================================================

def inspect_channel(channel_id, train, test, intervals):

    print("\n" + "=" * 65)
    print(f"CHANNEL: {channel_id}")
    print("=" * 65)

    if train is not None:
        print(f"Training shape: {train.shape}")
    else:
        print("Training file: Missing")

    if test is not None:
        print(f"Testing shape: {test.shape}")
    else:
        print("Testing file: Missing")

    print(f"Anomaly intervals: {len(intervals)}")

    for item in intervals:
        print(
            f"  [{item['start']}, {item['end']}] "
            f"Class: {item['class']}"
        )

    if train is not None:
        print(
            f"Training F0 mean: "
            f"{np.mean(train[:, TELEMETRY_FEATURE]):.6f}"
        )

    if test is not None:
        print(
            f"Testing F0 mean: "
            f"{np.mean(test[:, TELEMETRY_FEATURE]):.6f}"
        )


# ============================================================
# 5. TRAIN AND TEST TELEMETRY PLOT
# ============================================================

def plot_train_test(channel_id, train, test, intervals):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9)
    )

    if train is not None:

        train_signal = train[:, TELEMETRY_FEATURE]

        axes[0].plot(
            np.arange(len(train_signal)),
            train_signal,
            linewidth=0.7
        )

        axes[0].set_title(
            f"{channel_id}: Training Telemetry (F0)"
        )

        axes[0].set_ylabel("Telemetry value")
        axes[0].set_xlabel("Time index")
        axes[0].grid(alpha=0.3)

    else:
        axes[0].set_title(
            f"{channel_id}: Training data unavailable"
        )

    if test is not None:

        test_signal = test[:, TELEMETRY_FEATURE]

        axes[1].plot(
            np.arange(len(test_signal)),
            test_signal,
            linewidth=0.7,
            label="Test telemetry"
        )

        for i, interval in enumerate(intervals):

            start = interval["start"]
            end = interval["end"]

            # Only plot intervals within this sequence.
            if start >= len(test_signal):
                continue

            end = min(end, len(test_signal) - 1)

            color = (
                "red"
                if interval["class"] == "point"
                else "orange"
            )

            axes[1].axvspan(
                start,
                end,
                alpha=0.2,
                color=color,
                label=(
                    f"{interval['class']} anomaly"
                    if i == 0
                    else None
                )
            )

        axes[1].set_title(
            f"{channel_id}: Test Telemetry with Anomaly Labels"
        )

        axes[1].set_ylabel("Telemetry value")
        axes[1].set_xlabel("Time index")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

    else:
        axes[1].set_title(
            f"{channel_id}: Test data unavailable"
        )

    fig.tight_layout()

    output = PLOT_DIR / f"{channel_id}_train_test.png"

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output}")


# ============================================================
# 6. TELEMETRY DISTRIBUTION COMPARISON
# ============================================================

def plot_distributions(channel_id, train, test):

    if train is None or test is None:
        print(
            f"Skipping distribution plot for {channel_id}: "
            "train or test data missing."
        )
        return

    train_signal = train[:, TELEMETRY_FEATURE]
    test_signal = test[:, TELEMETRY_FEATURE]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    axes[0].hist(
        train_signal,
        bins=50,
        alpha=0.7
    )

    axes[0].set_title("Training F0 Distribution")
    axes[0].set_xlabel("Telemetry value")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.2)

    axes[1].hist(
        test_signal,
        bins=50,
        alpha=0.7
    )

    axes[1].set_title("Testing F0 Distribution")
    axes[1].set_xlabel("Telemetry value")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        f"{channel_id}: Training vs Testing Distribution"
    )

    fig.tight_layout()

    output = PLOT_DIR / f"{channel_id}_distribution.png"

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output}")


# ============================================================
# 7. CREATE CHANNEL PROFILE REPORT
# ============================================================

def create_profile_report(profile_rows):

    df = pd.DataFrame(profile_rows)

    output = REPORT_DIR / "representative_channel_profile.csv"

    df.to_csv(output, index=False)

    print("\nREPRESENTATIVE CHANNEL PROFILE")
    print("=" * 65)

    print(df.to_string(index=False))

    print(f"\nSaved: {output}")


# ============================================================
# 8. MAIN
# ============================================================

def main():

    labels = load_labels()

    profile_rows = []

    for channel_id in SELECTED_CHANNELS:

        train = load_channel(channel_id, TRAIN_DIR)
        test = load_channel(channel_id, TEST_DIR)

        intervals = get_channel_labels(
            labels,
            channel_id
        )

        if train is None and test is None:
            print(f"\nSkipping {channel_id}: files not found.")
            continue

        inspect_channel(
            channel_id,
            train,
            test,
            intervals
        )

        spacecraft = (
            intervals[0]["spacecraft"]
            if intervals
            else "Unlabelled"
        )

        profile_rows.append({
            "channel": channel_id,
            "spacecraft": spacecraft,
            "train_shape": (
                str(train.shape) if train is not None else None
            ),
            "test_shape": (
                str(test.shape) if test is not None else None
            ),
            "train_length": (
                len(train) if train is not None else None
            ),
            "test_length": (
                len(test) if test is not None else None
            ),
            "feature_count": (
                test.shape[1] if test is not None else None
            ),
            "anomaly_interval_count": len(intervals),
            "anomaly_classes": ", ".join(
                sorted(set(x["class"] for x in intervals))
            ),
        })

        plot_train_test(
            channel_id,
            train,
            test,
            intervals
        )

        plot_distributions(
            channel_id,
            train,
            test
        )

    create_profile_report(profile_rows)

    print("\n" + "=" * 65)
    print("REPRESENTATIVE CHANNEL EDA COMPLETED")
    print("=" * 65)

    print(f"Plots: {PLOT_DIR}")
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()
