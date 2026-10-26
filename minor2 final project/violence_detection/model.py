from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ViolenceDetector(nn.Module):
    def __init__(self, hidden_size: int = 256, dropout: float = 0.25):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal = nn.GRU(
            input_size=576,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = video.shape
        frames = video.reshape(batch * steps, channels, height, width)
        features = self.feature_extractor(frames)
        features = self.pool(features).flatten(1)
        features = features.reshape(batch, steps, -1)
        output, _ = self.temporal(features)
        logits = self.classifier(output[:, -1])
        return logits.squeeze(1)


def load_model(weights_path: str, device: torch.device) -> ViolenceDetector:
    model = ViolenceDetector()
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

