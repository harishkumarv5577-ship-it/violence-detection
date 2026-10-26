from __future__ import annotations

from pathlib import Path

import cv2
import torch


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


def list_video_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_video_frames(path: Path, max_frames: int | None = None) -> list:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"Video has no readable frames: {path}")
    return frames

