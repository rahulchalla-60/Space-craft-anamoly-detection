
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "test" / "P-2.npy"

OUTPUT_DIR = ROOT / "artifacts" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load telemetry
data = np.load(DATA_PATH)

# Command features: F1 to F24
commands = data[:, 1:]

# Inclusive interval definitions
regions = {
    "Before anomaly": (4800, 5299),
    "Annotation 2 only": (5300, 5349),
    "Common overlap": (5350, 6420),
    "Annotation 1 only": (6421, 6575),
    "After anomaly": (6700, 7100),
}


results = []

print("\nP-2 COMMAND ACTIVITY ANALYSIS")
print("=" * 75)

for region_name, (start, end) in regions.items():

    region_data = commands[start:end + 1]

    print(f"\n{region_name}: [{start}, {end}]")
    print("-" * 60)

    for feature_index in range(commands.shape[1]):

        feature_values = region_data[:, feature_index]

        active_count = np.count_nonzero(feature_values)
        total_count = len(feature_values)

        active_percentage = (
            active_count / total_count
        ) * 100

        results.append({
            "region": region_name,
            "feature": f"F{feature_index + 1}",
            "active_samples": active_count,
            "total_samples": total_count,
            "active_percentage": active_percentage,
        })

        print(
            f"F{feature_index + 1:02d}: "
            f"{active_count:4d}/{total_count:4d} "
            f"({active_percentage:6.2f}%)"
        )


# Save results
df = pd.DataFrame(results)

output_path = OUTPUT_DIR / "p2_command_activity.csv"

df.to_csv(output_path, index=False)

print("\n" + "=" * 75)
print(f"Analysis saved to: {output_path}")
print("Command activity analysis completed.")
