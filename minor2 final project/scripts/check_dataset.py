from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))

from violence_detection.utils import list_video_files


CLASSES = ("violence", "non_violence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check dataset folder counts and readable videos.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def inspect_video(path: Path) -> tuple[bool, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return False, 0, 0.0
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return frames > 0, frames, fps
    finally:
        capture.release()


def main() -> None:
    args = parse_args()
    total_errors = 0

    for split in ("train", "val"):
        print(f"\n{split}")
        print("-" * len(split))
        for class_name in CLASSES:
            folder = args.data_dir / split / class_name
            videos = list_video_files(folder) if folder.exists() else []
            unreadable = []
            short = []

            for video in videos:
                ok, frames, _fps = inspect_video(video)
                if not ok:
                    unreadable.append(video)
                elif frames < 16:
                    short.append(video)

            total_errors += len(unreadable)
            print(f"{class_name}: {len(videos)} videos")
            if unreadable:
                print(f"  unreadable: {len(unreadable)}")
                for path in unreadable[:5]:
                    print(f"  - {path}")
            if short:
                print(f"  warning: {len(short)} videos have fewer than 16 frames")

    if total_errors:
        raise SystemExit(f"\nDataset check failed: {total_errors} unreadable videos.")
    print("\nDataset check complete.")


if __name__ == "__main__":
    main()
