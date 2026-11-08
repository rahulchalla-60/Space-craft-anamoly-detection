
import pandas as pd
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

csv_path = ROOT / "data" / "labeled_anomalies.csv"

df = pd.read_csv(csv_path)

print("=" * 70)
print("ANOMALY LABEL INSPECTION")
print("=" * 70)

# 1. Basic information
print("\n1. DATASET INFORMATION")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# 2. First records
print("\n2. FIRST 10 RECORDS")
print(df.head(10).to_string())

# 3. Missing values
print("\n3. MISSING VALUES")
print(df.isnull().sum())

# 4. Spacecraft distribution
print("\n4. SPACECRAFT DISTRIBUTION")
if "spacecraft" in df.columns:
    print(df["spacecraft"].value_counts())

# 5. Anomaly class distribution
print("\n5. ANOMALY CLASS DISTRIBUTION")
if "class" in df.columns:
    print(df["class"].value_counts())

# 6. Parse anomaly intervals
def parse_intervals(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)

    if value is None:
        return []

    return value

print("\n6. ANOMALY INTERVAL INSPECTION")

if "anomaly_sequences" in df.columns:
    df["parsed_intervals"] = (
        df["anomaly_sequences"].apply(parse_intervals)
    )

    df["interval_count"] = (
        df["parsed_intervals"].apply(len)
    )

    print(
        df[
            ["chan_id", "class", "interval_count"]
        ].to_string(index=False)
    )

    print("\nTotal anomaly intervals:",
          df["interval_count"].sum())

# 7. Channel information
print("\n7. CHANNEL INFORMATION")

if "chan_id" in df.columns:
    print("Unique labeled channels:", df["chan_id"].nunique())

# 8. Anomaly interval examples
print("\n8. EXAMPLE ANOMALY INTERVALS")

for _, row in df.head(5).iterrows():
    print("\nChannel:", row["chan_id"])
    print("Class:", row.get("class"))
    print("Intervals:", row.get("parsed_intervals"))

# 9. Save cleaned inspection output
output = ROOT / "artifacts" / "reports"
output.mkdir(parents=True, exist_ok=True)

df.to_csv(
    output / "anomaly_label_inspection.csv",
    index=False
)

print("\nSaved anomaly label inspection.")
