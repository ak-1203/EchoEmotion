from __future__ import annotations

import argparse
import subprocess
from shutil import which
from pathlib import Path


def resolve_ffmpeg_bin(ffmpeg_bin: str | None) -> str:
    candidate = ffmpeg_bin or which("ffmpeg") or which("ffmpeg.exe")
    if not candidate:
        raise FileNotFoundError(
            "FFmpeg executable not found. Add ffmpeg to PATH or pass --ffmpeg-bin with the full executable path."
        )
    return candidate


def convert_file(source_path: Path, sample_rate: int, ffmpeg_bin: str) -> bool:
    target_path = source_path.with_suffix(".wav")
    if target_path.exists():
        return False

    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        str(target_path),
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed for {source_path} with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )

    return True


def convert_directory(dataset_root: Path, sample_rate: int, ffmpeg_bin: str) -> tuple[int, int]:
    converted = 0
    skipped = 0

    for source_path in sorted(dataset_root.rglob("*.m4a")):
        if convert_file(source_path, sample_rate, ffmpeg_bin):
            converted += 1
        else:
            skipped += 1

    return converted, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all .m4a files under the test dataset to 16 kHz mono .wav using FFmpeg."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent / "test_dataset",
        help="Root directory containing Actor* folders.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Output sample rate for converted wav files.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default=None,
        help="Optional full path to the FFmpeg executable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
    converted, skipped = convert_directory(dataset_root, args.sample_rate, ffmpeg_bin)
    print(f"Converted: {converted}")
    print(f"Skipped existing wav: {skipped}")


if __name__ == "__main__":
    main()
