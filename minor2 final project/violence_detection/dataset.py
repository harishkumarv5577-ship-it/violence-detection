from __future__ import annotations

import random
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from violence_detection.config import VideoConfig
from violence_detection.utils import list_video_files, read_video_frames


class ViolenceVideoDataset(Dataset):
    def __init__(self, split_dir: Path, config: VideoConfig, training: bool):
        self.config = config
        self.training = training
        self.samples = self._collect_samples(split_dir)
        self.transform = self._build_transform(training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        frames = read_video_frames(path)
        selected = self._sample_frames(frames)
        tensor = torch.stack([self.transform(self._to_pil(frame)) for frame in selected])
        return tensor, torch.tensor(label, dtype=torch.float32)

    def _collect_samples(self, split_dir: Path) -> list[tuple[Path, int]]:
        classes = {
            "non_violence": 0,
            "nonviolent": 0,
            "normal": 0,
            "violence": 1,
            "violent": 1,
        }
        samples: list[tuple[Path, int]] = []
        for class_name, label in classes.items():
            class_dir = split_dir / class_name
            if class_dir.exists():
                samples.extend((path, label) for path in list_video_files(class_dir))

        if not samples:
            raise ValueError(f"No videos found in {split_dir}. Expected violence/non_violence folders.")
        return samples

    def _sample_frames(self, frames: list) -> list:
        needed_span = (self.config.sequence_length - 1) * self.config.frame_stride + 1
        if len(frames) >= needed_span:
            max_start = len(frames) - needed_span
            start = random.randint(0, max_start) if self.training else max_start // 2
            indexes = [start + i * self.config.frame_stride for i in range(self.config.sequence_length)]
            return [frames[i] for i in indexes]

        indexes = [min(round(i * (len(frames) - 1) / max(self.config.sequence_length - 1, 1)), len(frames) - 1) for i in range(self.config.sequence_length)]
        return [frames[i] for i in indexes]

    def _to_pil(self, frame) -> Image.Image:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _build_transform(self, training: bool):
        base = [
            transforms.Resize((self.config.image_size, self.config.image_size)),
        ]
        if training:
            base.append(transforms.RandomHorizontalFlip(p=0.5))
        base.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transforms.Compose(base)

