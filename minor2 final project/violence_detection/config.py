from dataclasses import dataclass


@dataclass(frozen=True)
class VideoConfig:
    image_size: int = 224
    sequence_length: int = 16
    frame_stride: int = 2
    smoothing_window: int = 8
    violence_threshold: float = 0.65


@dataclass(frozen=True)
class TrainConfig(VideoConfig):
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0

