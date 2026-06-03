"""
Plot representative samples from the MRL Eye Dataset.

Rows:
    Male
    Female

Columns:
    Closed-NoGlasses
    Closed-Glasses
    Open-NoGlasses
    Open-Glasses
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mrlEyes_2018_01"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

FIGURES_DIR.mkdir(exist_ok=True)


def parse_filename(filename: str):

    name = filename.removesuffix(".png")
    parts = name.split("_")

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


def find_sample(image_paths, gender, eye_state, glasses):

    for path in image_paths:
        meta = parse_filename(path.name)

        if (
            meta["gender"] == gender
            and meta["eye_state"] == eye_state
            and meta["glasses"] == glasses
        ):
            return path

    return None


def main():

    image_paths = sorted(DATA_DIR.glob("**/*.png"))

    configs = [
        ("0", "Male"),
        ("1", "Female"),
    ]

    column_titles = [
        "Closed\nNo Glasses",
        "Closed\nGlasses",
        "Open\nNo Glasses",
        "Open\nGlasses",
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(12, 6),
    )

    for row, (gender, gender_name) in enumerate(configs):
        combinations = [
            ("0", "0"),  # closed, no glasses
            ("0", "1"),  # closed, glasses
            ("1", "0"),  # open, no glasses
            ("1", "1"),  # open, glasses
        ]

        for col, (eye_state, glasses) in enumerate(combinations):
            ax = axes[row][col]

            image_path = find_sample(
                image_paths,
                gender,
                eye_state,
                glasses,
            )

            if image_path is None:
                ax.text(
                    0.5,
                    0.5,
                    "No Sample",
                    ha="center",
                    va="center",
                )

                ax.axis("off")
                continue

            image = Image.open(image_path)

            ax.imshow(image, cmap="gray")
            ax.axis("off")

            if row == 0:
                ax.set_title(
                    column_titles[col],
                    fontsize=11,
                    fontweight="bold",
                )

            if col == 0:
                ax.set_ylabel(
                    gender_name,
                    fontsize=12,
                    fontweight="bold",
                    rotation=90,
                    labelpad=15,
                )

    plt.suptitle(
        "Representative Samples from the MRL Eye Dataset",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "representative_samples.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {FIGURES_DIR / 'representative_samples.png'}")


if __name__ == "__main__":
    main()
