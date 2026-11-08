
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

print("=" * 70)
print("NASA SMAP/MSL DATASET STRUCTURE INSPECTION")
print("=" * 70)

# 1. Main components
required = [
    DATA / "train",
    DATA / "test",
    DATA / "labeled_anomalies.csv",
]

print("\n1. REQUIRED COMPONENTS")

for path in required:
    print(
        f"{'[FOUND]' if path.exists() else '[MISSING]'} "
        f"{path.relative_to(ROOT)}"
    )

# 2. Training and testing files
train_files = sorted((DATA / "train").glob("*.npy"))
test_files = sorted((DATA / "test").glob("*.npy"))

train_names = {f.stem for f in train_files}
test_names = {f.stem for f in test_files}

print("\n2. TELEMETRY FILE COUNTS")
print("Training files:", len(train_files))
print("Testing files:", len(test_files))

print("\n3. CHANNEL MATCHING")

print("Channels in both:", len(train_names & test_names))
print("Only in training:", sorted(train_names - test_names))
print("Only in testing:", sorted(test_names - train_names))

# 3. File extensions
print("\n4. FILE EXTENSION DISTRIBUTION")

extensions = Counter(
    file.suffix.lower()
    for file in DATA.rglob("*")
    if file.is_file()
)

for extension, count in extensions.items():
    print(extension or "[NO EXTENSION]", count)

# 4. Additional artifacts
artifact_dir = DATA / "2018-05-19_15.00.10"

print("\n5. ADDITIONAL EXPERIMENT ARTIFACTS")

if artifact_dir.exists():
    for directory in sorted(artifact_dir.iterdir()):
        if directory.is_dir():
            files = list(directory.rglob("*"))
            file_count = sum(f.is_file() for f in files)

            print(f"\n{directory.name}/")
            print("Total files:", file_count)

            for file in [
                f for f in files if f.is_file()
            ][:10]:
                print("  ", file.relative_to(artifact_dir))

# 5. Final checks
print("\n6. BASIC VALIDATION")

print("Training/testing channel sets match:",
      train_names == test_names)

print("82 training files:", len(train_files) == 82)
print("82 testing files:", len(test_files) == 82)

print("\nInspection completed.")
