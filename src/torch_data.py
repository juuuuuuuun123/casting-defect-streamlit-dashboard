from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


class CastingImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(Path(row["path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image, label, row["relative_path"]


def build_transforms(
    image_size: int = 224,
    train: bool = True,
    augmentation: str = "weak",
    normalization: str = "imagenet",
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406] if normalization == "imagenet" else [0.5, 0.5, 0.5],
        std=[0.229, 0.224, 0.225] if normalization == "imagenet" else [0.5, 0.5, 0.5],
    )
    if not train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    steps = [transforms.Resize((image_size, image_size))]
    if augmentation == "weak":
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
    elif augmentation == "medium":
        steps.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    steps.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(steps)


def make_weighted_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
    counts = frame["label"].value_counts().to_dict()
    weights = frame["label"].map(lambda label: 1.0 / counts[int(label)]).to_numpy()
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)
