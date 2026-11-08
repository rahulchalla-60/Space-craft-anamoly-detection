
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS = ROOT / "data" / "2018-05-19_15.00.10"

print("=" * 70)
print("EXISTING EXPERIMENT ARTIFACT INSPECTION")
print("=" * 70)

if not ARTIFACTS.exists():
    raise FileNotFoundError(ARTIFACTS)

for folder in sorted(ARTIFACTS.iterdir()):

    if not folder.is_dir():
        continue

    print(f"\n{'=' * 50}")
    print("DIRECTORY:", folder.name)
    print("=" * 50)

    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file()
    )

    print("Total files:", len(files))

    for file in files:
        print("\nFile:", file.relative_to(ARTIFACTS))
        print("Extension:", file.suffix or "No extension")
        print("Size:", file.stat().st_size, "bytes")

        if file.suffix == ".npy":
            try:
                import numpy as np

                arr = np.load(
                    file,
                    mmap_mode="r",
                    allow_pickle=False
                )

                print("Shape:", arr.shape)
                print("Dtype:", arr.dtype)

            except Exception as error:
                print("Could not inspect NPY:", error)

print("\nArtifact inspection completed.")
