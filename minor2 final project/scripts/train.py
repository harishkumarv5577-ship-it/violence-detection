from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from violence_detection.config import TrainConfig
from violence_detection.dataset import ViolenceVideoDataset
from violence_detection.model import ViolenceDetector
from violence_detection.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a real-time violence detector.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    losses = []
    labels = []
    predictions = []

    for videos, targets in tqdm(loader, leave=False):
        videos = videos.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(training):
            logits = model(videos)
            loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        probabilities = torch.sigmoid(logits).detach().cpu()
        losses.append(loss.item())
        labels.extend(targets.cpu().int().tolist())
        predictions.extend((probabilities >= 0.5).int().tolist())

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        sequence_length=args.sequence_length,
        frame_stride=args.frame_stride,
        learning_rate=args.learning_rate,
    )
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ViolenceVideoDataset(args.data_dir / "train", config, training=True)
    val_dataset = ViolenceVideoDataset(args.data_dir / "val", config, training=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=config.num_workers)

    model = ViolenceDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, training=False)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "val_metrics": val_metrics,
                },
                args.output_dir / "best.pt",
            )


if __name__ == "__main__":
    main()
