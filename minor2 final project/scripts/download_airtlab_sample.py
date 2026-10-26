from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


API_ROOT = "https://api.github.com/repos/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/contents"
DATASET_ROOT = "violence-detection-dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small balanced sample from the public AIRTLab dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--camera", choices=("cam1", "cam2"), default="cam1")
    return parser.parse_args()


def read_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "violence-detection-starter"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"exists: {destination}")
        return
    request = urllib.request.Request(url, headers={"User-Agent": "violence-detection-starter"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())
    print(f"downloaded: {destination}")


def list_remote_files(label_dir: str, camera: str) -> list[dict]:
    path = f"{DATASET_ROOT}/{label_dir}/{camera}"
    entries = read_json(f"{API_ROOT}/{path}")
    return sorted(
        [entry for entry in entries if entry["type"] == "file" and entry["name"].lower().endswith(".mp4")],
        key=lambda entry: entry["name"],
    )


def main() -> None:
    args = parse_args()
    mapping = {
        "violent": "violence",
        "non-violent": "non_violence",
    }

    for remote_label, local_label in mapping.items():
        entries = list_remote_files(remote_label, args.camera)
        selected = entries[: args.per_class]
        if len(selected) < args.per_class:
            raise SystemExit(f"Only found {len(selected)} files for {remote_label}/{args.camera}.")

        for entry in selected:
            destination = args.output_dir / local_label / f"airtlab_{args.camera}_{entry['name']}"
            download_file(entry["download_url"], destination)


if __name__ == "__main__":
    main()

