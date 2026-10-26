from __future__ import annotations

from collections import deque

import cv2
import torch
from PIL import Image
from torchvision import transforms

from violence_detection.config import VideoConfig


class RealTimePredictor:
    def __init__(self, model: torch.nn.Module, device: torch.device, config: VideoConfig):
        self.model = model
        self.device = device
        self.config = config
        self.frames = deque(maxlen=config.sequence_length)
        self.probabilities = deque(maxlen=config.smoothing_window)
        self.transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def update(self, frame) -> tuple[str, float] | None:
        self.frames.append(frame.copy())
        if len(self.frames) < self.config.sequence_length:
            return None

        video = torch.stack([self.transform(self._to_pil(item)) for item in self.frames])
        video = video.unsqueeze(0).to(self.device)
        logit = self.model(video)
        probability = torch.sigmoid(logit).item()
        self.probabilities.append(probability)
        smoothed = sum(self.probabilities) / len(self.probabilities)
        label = "VIOLENCE" if smoothed >= self.config.violence_threshold else "NORMAL"
        return label, smoothed

    def _to_pil(self, frame) -> Image.Image:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


def draw_prediction(frame, prediction: tuple[str, float] | None, threshold: float):
    if prediction is None:
        text = "Warming up..."
        color = (0, 200, 255)
    else:
        label, probability = prediction
        text = f"{label}  p={probability:.2f}  threshold={threshold:.2f}"
        color = (0, 0, 255) if label == "VIOLENCE" else (0, 180, 0)

    cv2.rectangle(frame, (12, 12), (620, 58), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    return frame

