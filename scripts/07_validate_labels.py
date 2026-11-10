
import ast
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
TEST = DATA / "test"
CSV_PATH = DATA / "labeled_anomalies.csv"

df = pd.read_csv(CSV_PATH)


# --------------------------------------------------
# 2. PARSE ANOMALY INTERVALS AND CLASSES
# --------------------------------------------------
def parse_classes(value):
    """
    Convert dataset class strings into Python lists.

    Example:
        '[point, contextual]'
        becomes:
        ['point', 'contextual']
    """
    value = str(value).strip()

    # Remove surrounding square brackets
    value = value.strip("[]")

    if not value:
        return []

    # Split by commas and remove whitespace
    return [
        item.strip().strip("'\"")
        for item in value.split(",")
    ]
def parse_list(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


df["intervals"] = df["anomaly_sequences"].apply(parse_list)
df["classes"] = df["class"].apply(parse_classes)


# --------------------------------------------------
# 3. VALIDATION COUNTERS
# --------------------------------------------------

issues = []

def report_issue(category, channel, details):
    issues.append({
        "category": category,
        "channel": channel,
        "details": details
    })


print("=" * 75)
print("NASA SMAP/MSL LABEL INTEGRITY VALIDATION")
print("=" * 75)


# --------------------------------------------------
# 4. DUPLICATE CHANNEL ID VALIDATION
# --------------------------------------------------

print("\n1. DUPLICATE CHANNEL ID VALIDATION")

duplicates = df[df.duplicated("chan_id", keep=False)]

if duplicates.empty:
    print("No duplicate channel IDs found.")
else:
    print("Duplicate channel IDs found:")
    print(
        duplicates[
            ["chan_id", "spacecraft", "class", "num_values"]
        ].to_string(index=False)
    )

    for channel in duplicates["chan_id"].unique():
        report_issue(
            "Duplicate channel ID",
            channel,
            "Channel ID occurs in multiple CSV records."
        )


# --------------------------------------------------
# 5. DUPLICATE COMPOSITE KEY VALIDATION
# --------------------------------------------------

print("\n2. DUPLICATE SPACECRAFT + CHANNEL VALIDATION")

duplicate_keys = df[
    df.duplicated(
        subset=["spacecraft", "chan_id"],
        keep=False
    )
]

if duplicate_keys.empty:
    print("No duplicate composite keys found.")
else:
    print("Duplicate composite keys found:")
    print(
        duplicate_keys[
            ["spacecraft", "chan_id", "class"]
        ].to_string(index=False)
    )


# --------------------------------------------------
# 6. CLASS AND INTERVAL ALIGNMENT
# --------------------------------------------------

print("\n3. CLASS AND INTERVAL ALIGNMENT")

alignment_errors = 0

for _, row in df.iterrows():

    channel = row["chan_id"]

    intervals = row["intervals"]
    classes = row["classes"]

    if len(intervals) != len(classes):

        alignment_errors += 1

        report_issue(
            "Class-interval mismatch",
            channel,
            f"{len(intervals)} intervals but "
            f"{len(classes)} classes."
        )

        print(
            f"MISMATCH: {channel} | "
            f"Intervals={len(intervals)} | "
            f"Classes={len(classes)}"
        )

if alignment_errors == 0:
    print("All records have matching interval/class counts.")


# --------------------------------------------------
# 7. TEST SEQUENCE LENGTH VALIDATION
# --------------------------------------------------

print("\n4. TEST SEQUENCE LENGTH VALIDATION")

length_errors = 0

for _, row in df.iterrows():

    channel = row["chan_id"]

    test_path = TEST / f"{channel}.npy"

    if not test_path.exists():

        length_errors += 1

        report_issue(
            "Missing test file",
            channel,
            str(test_path)
        )

        print("MISSING TEST FILE:", channel)
        continue

    data = np.load(
        test_path,
        mmap_mode="r",
        allow_pickle=False
    )

    actual_length = data.shape[0]
    expected_length = int(row["num_values"])

    if actual_length != expected_length:

        length_errors += 1

        report_issue(
            "Sequence length mismatch",
            channel,
            f"CSV={expected_length}, "
            f"Test={actual_length}"
        )

        print(
            f"LENGTH MISMATCH: {channel} | "
            f"CSV={expected_length} | "
            f"TEST={actual_length}"
        )

if length_errors == 0:
    print("All CSV sequence lengths match test files.")


# --------------------------------------------------
# 8. ANOMALY INTERVAL BOUNDARY VALIDATION
# --------------------------------------------------

print("\n5. ANOMALY INTERVAL BOUNDARY VALIDATION")

invalid_intervals = 0
valid_intervals = 0

for _, row in df.iterrows():

    channel = row["chan_id"]

    test_path = TEST / f"{channel}.npy"

    if not test_path.exists():
        continue

    data = np.load(
        test_path,
        mmap_mode="r",
        allow_pickle=False
    )

    sequence_length = data.shape[0]

    for interval, anomaly_class in zip(
        row["intervals"],
        row["classes"]
    ):

        if len(interval) != 2:
            invalid_intervals += 1

            report_issue(
                "Invalid interval format",
                channel,
                str(interval)
            )
            continue

        start, end = map(int, interval)

        if (
            start < 0
            or end < start
            or end >= sequence_length
        ):

            invalid_intervals += 1

            report_issue(
                "Invalid anomaly interval",
                channel,
                f"Interval={interval}, "
                f"Sequence length={sequence_length}"
            )

            print(
                f"INVALID: {channel} | "
                f"Interval={interval} | "
                f"Length={sequence_length}"
            )

        else:
            valid_intervals += 1

print("Valid intervals:", valid_intervals)
print("Invalid intervals:", invalid_intervals)


# --------------------------------------------------
# 9. CLASS DISTRIBUTION
# --------------------------------------------------

print("\n6. ANOMALY CLASS DISTRIBUTION")

class_counts = {
    "point": 0,
    "contextual": 0
}

for _, row in df.iterrows():

    for anomaly_class in row["classes"]:

        if anomaly_class in class_counts:
            class_counts[anomaly_class] += 1
        else:
            report_issue(
                "Unknown anomaly class",
                row["chan_id"],
                str(anomaly_class)
            )

print("Point anomalies:", class_counts["point"])
print("Contextual anomalies:", class_counts["contextual"])

total_classes = sum(class_counts.values())

print("Total labeled intervals:", total_classes)


# --------------------------------------------------
# 10. CHANNEL IDENTITY AND DUPLICATE INVESTIGATION
# --------------------------------------------------

print("\n7. DUPLICATE CHANNEL INVESTIGATION")

for channel in duplicates["chan_id"].unique():

    records = df[df["chan_id"] == channel]

    print(f"\nChannel: {channel}")

    for _, row in records.iterrows():

        print("Spacecraft:", row["spacecraft"])
        print("Num values:", row["num_values"])
        print("Intervals:", row["intervals"])
        print("Classes:", row["classes"])


# --------------------------------------------------
# 11. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 75)
print("VALIDATION SUMMARY")
print("=" * 75)

print("Total CSV records:", len(df))
print("Unique channel IDs:", df["chan_id"].nunique())
print(
    "Unique spacecraft-channel pairs:",
    df[["spacecraft", "chan_id"]].drop_duplicates().shape[0]
)

print("Class alignment errors:", alignment_errors)
print("Sequence length errors:", length_errors)
print("Valid anomaly intervals:", valid_intervals)
print("Invalid anomaly intervals:", invalid_intervals)
print("Total validation issues:", len(issues))


# --------------------------------------------------
# 12. SAVE VALIDATION REPORT
# --------------------------------------------------

output_dir = ROOT / "artifacts" / "reports"
output_dir.mkdir(parents=True, exist_ok=True)

issues_df = pd.DataFrame(
    issues,
    columns=["category", "channel", "details"]
)

report_path = output_dir / "label_validation_issues.csv"

issues_df.to_csv(report_path, index=False)

print("\nValidation issues saved to:")
print(report_path)

print("\nValidation completed.")
