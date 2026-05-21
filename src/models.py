from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


def _replace_classifier(model: nn.Module, model_name: str) -> int:
    if model_name == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        return in_features
    if model_name == "mobilenet_v3_small":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return in_features
    if model_name == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return in_features
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_model(model_name: str = "small_cnn", pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    if model_name == "small_cnn":
        return SmallCNN()

    weights = "DEFAULT" if pretrained else None
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    _replace_classifier(model, model_name)
    for parameter in model.parameters():
        if parameter.ndim <= 1:
            parameter.requires_grad = True
    return model


def build_feature_extractor(model_name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    if model_name != "resnet18":
        raise ValueError("Embedding experiments currently use resnet18 features for a stable feature size.")
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    return model
