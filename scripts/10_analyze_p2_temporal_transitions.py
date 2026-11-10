
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "test" / "P-2.npy"

REPORT_DIR = ROOT / "artifacts" / "reports"
PLOT_DIR = ROOT / "artifacts" / "plots"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_ID = "P-2"

# Telemetry feature
TELEMETRY_FEATURE = 0

# Command features: F1-F24
COMMAND_START = 1

# Anomaly annotations
ANNOTATION_1 = (5350, 6575)
ANNOTATION_2 = (5300, 6420)

# Rolling standard deviation window
ROLLING_WINDOW = 100

# A telemetry spike is defined here as F0 > 0.
# This is an exploratory threshold, not a benchmark threshold.
TELEMETRY_SPIKE_THRESHOLD = 0.0

# Regions use inclusive endpoints.
REGIONS = {
    "Before anomaly": (4800, 5299),
    "Annotation 2 only": (5300, 5349),
    "Common overlap": (5350, 6420),
    "Annotation 1 only": (6421, 6575),
    "Unexamined interval": (6576, 6699),
    "After anomaly": (6700, 7100),
}


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    data = np.load(DATA_PATH)

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D data, received {data.shape}"
        )

    if data.shape[1] < 2:
        raise ValueError(
            "No command features found."
        )

    print("\nP-2 TEMPORAL TRANSITION ANALYSIS")
    print("=" * 70)

    print(f"Channel: {CHANNEL_ID}")
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Telemetry feature: F0")
    print(f"Command features: F1-F{data.shape[1] - 1}")

    return data


# ============================================================
# 3. VALIDATE COMMAND FEATURES
# ============================================================

def validate_commands(data):

    commands = data[:, COMMAND_START:]

    print("\nCOMMAND VALUE VALIDATION")
    print("=" * 70)

    for i in range(commands.shape[1]):

        feature = commands[:, i]
        unique_values = np.unique(feature)

        print(
            f"F{i + 1:02d}: "
            f"unique values = {unique_values[:10]}"
        )

        # This analysis assumes binary command features.
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(
                f"F{i + 1} is not binary. "
                f"Observed values: {unique_values}"
            )

    return commands


# ============================================================
# 4. COMMAND TRANSITIONS
# ============================================================

def analyze_transitions(commands):

    transition_results = []
    event_results = []

    print("\nCOMMAND TRANSITION ANALYSIS")
    print("=" * 70)

    for feature_index in range(commands.shape[1]):

        feature_name = f"F{feature_index + 1}"

        values = commands[:, feature_index]

        # Difference between consecutive samples.
        differences = np.diff(values)

        # Transition indices refer to the index of the new state.
        transition_indices = np.where(differences != 0)[0] + 1

        zero_to_one = np.where(differences == 1)[0] + 1
        one_to_zero = np.where(differences == -1)[0] + 1

        print(f"\n{feature_name}")
        print(f"Total transitions: {len(transition_indices)}")
        print(f"0 -> 1: {len(zero_to_one)}")
        print(f"1 -> 0: {len(one_to_zero)}")

        for index in transition_indices:

            previous_value = int(values[index - 1])
            current_value = int(values[index])

            event_results.append({
                "feature": feature_name,
                "time_index": int(index),
                "previous_value": previous_value,
                "current_value": current_value,
                "transition": (
                    f"{previous_value}->{current_value}"
                ),
            })

        for region_name, (start, end) in REGIONS.items():

            region_transitions = transition_indices[
                (transition_indices >= start)
                & (transition_indices <= end)
            ]

            region_01 = zero_to_one[
                (zero_to_one >= start)
                & (zero_to_one <= end)
            ]

            region_10 = one_to_zero[
                (one_to_zero >= start)
                & (one_to_zero <= end)
            ]

            transition_results.append({
                "feature": feature_name,
                "region": region_name,
                "total_transitions": len(region_transitions),
                "zero_to_one": len(region_01),
                "one_to_zero": len(region_10),
            })

    transitions_df = pd.DataFrame(transition_results)
    events_df = pd.DataFrame(event_results)

    transitions_df.to_csv(
        REPORT_DIR / "p2_command_transition_counts.csv",
        index=False
    )

    events_df.to_csv(
        REPORT_DIR / "p2_command_transition_events.csv",
        index=False
    )

    print("\nTransition reports saved.")

    return transitions_df, events_df


# ============================================================
# 5. ROLLING TELEMETRY STANDARD DEVIATION
# ============================================================

def calculate_rolling_statistics(data):

    telemetry = data[:, TELEMETRY_FEATURE]

    series = pd.Series(telemetry)

    rolling_mean = series.rolling(
        window=ROLLING_WINDOW,
        min_periods=ROLLING_WINDOW
    ).mean()

    rolling_std = series.rolling(
        window=ROLLING_WINDOW,
        min_periods=ROLLING_WINDOW
    ).std()

    rolling_df = pd.DataFrame({
        "time_index": np.arange(len(telemetry)),
        "telemetry": telemetry,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
    })

    output = REPORT_DIR / "p2_rolling_telemetry_statistics.csv"

    rolling_df.to_csv(output, index=False)

    print("\nROLLING TELEMETRY STATISTICS")
    print("=" * 70)

    print(f"Rolling window: {ROLLING_WINDOW}")
    print(f"Mean rolling std: {rolling_std.mean():.6f}")
    print(f"Maximum rolling std: {rolling_std.max():.6f}")
    print(
        "Maximum rolling std index:",
        rolling_std.idxmax()
    )

    return rolling_df


# ============================================================
# 6. TELEMETRY SPIKES AND COMMAND TRANSITION TIMING
# ============================================================

def analyze_spike_relationship(data, events_df):

    telemetry = data[:, TELEMETRY_FEATURE]

    spike_indices = np.where(
        telemetry > TELEMETRY_SPIKE_THRESHOLD
    )[0]

    print("\nTELEMETRY SPIKE ANALYSIS")
    print("=" * 70)

    print(
        f"Spike definition: F0 > "
        f"{TELEMETRY_SPIKE_THRESHOLD}"
    )

    print(f"Total spike samples: {len(spike_indices)}")

    print("\nSPIKES BY REGION")

    for region_name, (start, end) in REGIONS.items():

        count = np.sum(
            (spike_indices >= start)
            & (spike_indices <= end)
        )

        print(
            f"{region_name}: {count} spike samples"
        )

    if events_df.empty:
        return pd.DataFrame()

    rows = []

    # For each telemetry spike, find nearest command transition.
    transition_indices = events_df["time_index"].to_numpy()

    for spike_index in spike_indices:

        distances = transition_indices - spike_index

        nearest_position = np.argmin(np.abs(distances))

        nearest_event = events_df.iloc[nearest_position]

        rows.append({
            "spike_index": int(spike_index),
            "nearest_transition_index": int(
                nearest_event["time_index"]
            ),
            "feature": nearest_event["feature"],
            "transition": nearest_event["transition"],
            "time_difference": int(
                nearest_event["time_index"] - spike_index
            ),
        })

    result = pd.DataFrame(rows)

    output = REPORT_DIR / "p2_spike_command_relationship.csv"

    result.to_csv(output, index=False)

    print("\nNearest command-transition timing:")
    print(
        result["time_difference"].describe()
    )

    print(f"\nSaved: {output}")

    return result


# ============================================================
# 7. ANALYZE THE UNEXAMINED REGION
# ============================================================

def inspect_unexamined_region(data, commands):

    start, end = REGIONS["Unexamined interval"]

    telemetry = data[start:end + 1, TELEMETRY_FEATURE]

    command_region = commands[start:end + 1]

    print("\nUNEXAMINED REGION INSPECTION")
    print("=" * 70)

    print(f"Region: [{start}, {end}]")
    print(f"Number of samples: {len(telemetry)}")

    print("\nTelemetry:")
    print(f"Mean: {np.mean(telemetry):.6f}")
    print(f"Std: {np.std(telemetry):.6f}")
    print(f"Min: {np.min(telemetry):.6f}")
    print(f"Max: {np.max(telemetry):.6f}")

    print("\nCommand activity:")

    for i in range(command_region.shape[1]):

        active = np.count_nonzero(
            command_region[:, i]
        )

        print(
            f"F{i + 1:02d}: "
            f"{active}/{len(command_region)}"
        )

    print(
        "\nTotal nonzero command entries:",
        np.count_nonzero(command_region)
    )


# ============================================================
# 8. VISUALIZE ROLLING STANDARD DEVIATION
# ============================================================

def plot_rolling_statistics(rolling_df):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True
    )

    time = rolling_df["time_index"]

    axes[0].plot(
        time,
        rolling_df["telemetry"],
        linewidth=0.6,
        label="Telemetry F0"
    )

    axes[0].set_ylabel("Telemetry")
    axes[0].set_title("P-2 Telemetry and Rolling Variability")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        time,
        rolling_df["rolling_std"],
        linewidth=1,
        label=f"Rolling standard deviation ({ROLLING_WINDOW} samples)"
    )

    axes[1].set_xlabel("Time index")
    axes[1].set_ylabel("Rolling std")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    for ax in axes:

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

    fig.tight_layout()

    output = PLOT_DIR / "p2_rolling_telemetry_std.png"

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nSaved: {output}")


# ============================================================
# 9. VISUALIZE F5 AND F6 TRANSITIONS
# ============================================================

def plot_f5_f6(data):

    time = np.arange(len(data))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 7),
        sharex=True
    )

    for ax, feature_index in zip(axes, [5, 6]):

        ax.step(
            time,
            data[:, feature_index],
            where="post",
            linewidth=0.8
        )

        ax.set_ylabel(f"F{feature_index}")
        ax.set_title(
            f"P-2 Command Feature F{feature_index}"
        )

        ax.grid(alpha=0.3)

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

    axes[-1].set_xlabel("Time index")

    fig.tight_layout()

    output = PLOT_DIR / "p2_f5_f6_transitions.png"

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output}")


# ============================================================
# 10. MAIN
# ============================================================

def main():

    data = load_data()

    commands = validate_commands(data)

    transitions_df, events_df = analyze_transitions(
        commands
    )

    rolling_df = calculate_rolling_statistics(data)

    analyze_spike_relationship(
        data,
        events_df
    )

    inspect_unexamined_region(
        data,
        commands
    )

    plot_rolling_statistics(rolling_df)

    plot_f5_f6(data)

    print("\n" + "=" * 70)
    print("P-2 TEMPORAL ANALYSIS COMPLETED")
    print("=" * 70)

    print(f"Reports: {REPORT_DIR}")
    print(f"Plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
