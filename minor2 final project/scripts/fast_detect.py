from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultra-fast video scan for web demos.")
    parser.add_argument("--weights", help="Accepted for compatibility; not used in fast demo mode.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--threshold", type=float, default=6.5)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--max-scan-frames", type=int, default=40)
    parser.add_argument("--json", action="store_true", help="Print JSON for website/backend integration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(str(Path(args.source)))
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    previous = None
    scores = []
    scanned = 0
    step = 3

    try:
        while scanned < args.max_scan_frames:
            ok, frame = capture.read()
            if not ok:
                break
            scanned += 1
            if scanned % step != 0:
                continue

            frame = cv2.resize(frame, (args.image_size, args.image_size))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if previous is not None:
                diff = cv2.absdiff(previous, gray)
                scores.append(float(np.mean(diff)))
            previous = gray
    finally:
        capture.release()

    motion_score = sum(scores) / len(scores) if scores else 0.0
    confidence = min(max(motion_score / 14.0, 0.05), 0.95)
    label = "VIOLENCE" if motion_score >= args.threshold else "NORMAL"
    if args.json:
        print(json.dumps({"label": label, "confidence": round(confidence, 4), "motionScore": round(motion_score, 4)}))
    elif label == "VIOLENCE":
        print("Violence Detected")
    else:
        print("No Violence Detected")


if __name__ == "__main__":
    main()
