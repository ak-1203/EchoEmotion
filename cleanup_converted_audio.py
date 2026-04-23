from __future__ import annotations

import argparse
from pathlib import Path


def cleanup_directory(dataset_root: Path) -> tuple[int, int]:
    deleted = 0
    skipped = 0

    for source_path in sorted(dataset_root.rglob("*.m4a")):
        wav_path = source_path.with_suffix(".wav")
        if wav_path.exists():
            source_path.unlink()
            deleted += 1
        else:
            skipped += 1

    return deleted, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete .m4a files only when a same-name .wav file already exists."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent / "test_dataset",
        help="Root directory containing Actor* folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    deleted, skipped = cleanup_directory(dataset_root)
    print(f"Deleted .m4a files: {deleted}")
    print(f"Skipped without matching .wav: {skipped}")


if __name__ == "__main__":
    main()
