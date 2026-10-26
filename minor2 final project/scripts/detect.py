from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))

from violence_detection.config import VideoConfig
from violence_detection.model import load_model
from violence_detection.realtime import RealTimePredictor, draw_prediction
from violence_detection.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time violence detection.")
    parser.add_argument("--weights", required=True, help="Path to a trained .pt checkpoint.")
    parser.add_argument("--source", default="0", help="Camera index, RTSP URL, or video path.")
    parser.add_argument("--output", type=Path, help="Optional path to save an annotated video.")
    parser.add_argument("--no-display", action="store_true", help="Process without opening the webcam/video window.")
    parser.add_argument("--max-frames", type=int, help="Optional maximum number of frames to process.")
    parser.add_argument("--api-url", help="Optional web app URL for storing the final result, for example http://localhost:8080.")
    parser.add_argument("--email", help="Email/Gmail of the signed-in user for result storage.")
    parser.add_argument("--video-name", help="Optional display name for the stored result.")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--sequence-length", type=int, default=16)
    return parser.parse_args()


def normalize_source(source: str):
    return int(source) if source.isdigit() else source


def store_result(api_url: str, email: str, video_name: str, prediction: tuple[str, float], output_path: Path | None) -> None:
    label, confidence = prediction
    payload = {
        "email": email,
        "videoName": video_name,
        "detectionResult": label,
        "confidenceScore": f"{confidence:.4f}",
        "outputVideoPath": str(output_path) if output_path else "",
    }
    request = urllib.request.Request(
        url=f"{api_url.rstrip('/')}/api/results",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
        print("Stored detection result in H2 web app.")
    except urllib.error.URLError as error:
        print(f"Could not store detection result: {error}")


def main() -> None:
    args = parse_args()
    config = VideoConfig(sequence_length=args.sequence_length, violence_threshold=args.threshold)
    device = get_device()
    model = load_model(args.weights, device)
    predictor = RealTimePredictor(model, device, config)

    capture = cv2.VideoCapture(normalize_source(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    try:
        frame_count = 0
        final_prediction = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            prediction = predictor.update(frame)
            if prediction is not None:
                final_prediction = prediction
            frame = draw_prediction(frame, prediction, config.violence_threshold)
            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Real-Time Violence Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_count += 1
            if args.max_frames is not None and frame_count >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if args.api_url and args.email and final_prediction is not None:
        video_name = args.video_name or (Path(args.source).name if not args.source.isdigit() else f"camera-{args.source}")
        store_result(args.api_url, args.email, video_name, final_prediction, args.output)


if __name__ == "__main__":
    main()
