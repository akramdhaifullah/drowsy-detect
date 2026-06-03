"""
Exploratory Data Analysis for MRL Eye Dataset.

Parses filenames from the MRL Eye Dataset to extract annotation
statistics. Filename format:
    s{subject}_{imageID}_{gender}_{glasses}_{eyeState}_{reflections}_{lighting}_{sensor}.png

Usage:
    python utils/explore_dataset.py
"""

from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mrlEyes_2018_01"

LABELS = {
    "gender": {"0": "Laki-laki", "1": "Perempuan"},
    "glasses": {"0": "Tanpa Kacamata", "1": "Dengan Kacamata"},
    "eye_state": {"0": "Tertutup (Closed)", "1": "Terbuka (Open)"},
    "reflections": {"0": "Tidak Ada", "1": "Kecil", "2": "Besar"},
    "lighting": {"0": "Buruk (Bad)", "1": "Baik (Good)"},
    "sensor": {
        "01": "Intel RealSense SR300 (640x480)",
        "02": "IDS Imaging (1280x1024)",
        "03": "Aptina Imaging (752x480)",
    },
}


def parse_filename(filename: str) -> dict | None:
    """Parse a single MRL Eye Dataset filename into its annotation fields."""
    name = filename.removesuffix(".png")
    parts = name.split("_")
    if len(parts) != 8:
        return None
    return {
        "subject": parts[0],
        "image_id": parts[1],
        "gender": parts[2],
        "glasses": parts[3],
        "eye_state": parts[4],
        "reflections": parts[5],
        "lighting": parts[6],
        "sensor": parts[7],
    }


def main():
    if not DATA_DIR.exists():
        print(f"Error: Dataset directory not found at {DATA_DIR}")
        return

    # Collect all png files
    png_files = []
    subject_dirs = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir()], key=lambda x: x.name
    )

    for sdir in subject_dirs:
        for f in sdir.iterdir():
            if f.suffix == ".png":
                png_files.append(f.name)

    total = len(png_files)
    print(f"{'=' * 60}")
    print("MRL Eye Dataset - Exploratory Data Analysis")
    print(f"{'=' * 60}")
    print(f"Direktori : {DATA_DIR}")
    print(f"Total Subjek : {len(subject_dirs)}")
    print(f"Total Citra : {total:,}")
    print()

    # Parse all filenames
    counters = {
        "gender": Counter(),
        "glasses": Counter(),
        "eye_state": Counter(),
        "reflections": Counter(),
        "lighting": Counter(),
        "sensor": Counter(),
    }
    per_subject = Counter()

    for fname in png_files:
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        for key in counters:
            counters[key][parsed[key]] += 1
        per_subject[parsed["subject"]] += 1

    # Print distributions
    sections = [
        ("Kondisi Mata (Eye State)", "eye_state"),
        ("Gender", "gender"),
        ("Kacamata (Glasses)", "glasses"),
        ("Refleksi (Reflections)", "reflections"),
        ("Pencahayaan (Lighting)", "lighting"),
        ("Sensor", "sensor"),
    ]

    for title, key in sections:
        print(f"--- {title} ---")
        for val, count in sorted(counters[key].items()):
            label = LABELS[key].get(val, val)
            pct = count / total * 100
            print(f"  {val} ({label}): {count:>8,}  ({pct:5.1f}%)")
        print()

    # Per-subject summary
    print("--- Distribusi per Subjek ---")
    for subj, count in sorted(per_subject.items()):
        print(f"  {subj}: {count:>6,} citra")
    print()

    # Summary stats
    counts = list(per_subject.values())
    print("--- Statistik per Subjek ---")
    print(f"  Min  : {min(counts):,}")
    print(f"  Max  : {max(counts):,}")
    print(f"  Mean : {sum(counts) / len(counts):,.1f}")
    print()


if __name__ == "__main__":
    main()
