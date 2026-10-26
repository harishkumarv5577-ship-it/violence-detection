from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from violence_detection.utils import list_video_files


CLASSES = ("violence", "non_violence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw violence/non-violence videos into train and val folders.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them.")
    parser.add_argument("--reset", action="store_true", help="Remove existing split videos before creating a new split.")
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> None:
    for split in ("train", "val"):
        for class_name in CLASSES:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def reset_split_dirs(output_dir: Path) -> None:
    for split in ("train", "val"):
        for class_name in CLASSES:
            folder = output_dir / split / class_name
            if not folder.exists():
                continue
            for video in list_video_files(folder):
                video.unlink()


def transfer_file(source: Path, destination: Path, copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if copy:
        shutil.copy2(source, destination)
    else:
        shutil.move(str(source), str(destination))


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be between 0 and 1.")

    random.seed(args.seed)
    ensure_dirs(args.output_dir)
    if args.reset:
        reset_split_dirs(args.output_dir)

    for class_name in CLASSES:
        class_dir = args.raw_dir / class_name
        videos = list_video_files(class_dir) if class_dir.exists() else []
        if not videos:
            print(f"No videos found for {class_name}: {class_dir}")
            continue

        random.shuffle(videos)
        val_count = max(1, round(len(videos) * args.val_ratio)) if len(videos) > 1 else 0
        val_videos = set(videos[:val_count])

        for video in videos:
            split = "val" if video in val_videos else "train"
            destination = args.output_dir / split / class_name / video.name
            transfer_file(video, destination, copy=args.copy)

        print(
            f"{class_name}: {len(videos) - val_count} train, {val_count} val "
            f"({'copied' if args.copy else 'moved'})"
        )


if __name__ == "__main__":
    main()
