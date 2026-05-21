from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_folds: int = 5
    holdout_size: float = 0.15
    positive_label: str = "def_front"
    negative_label: str = "ok_front"


def ensure_project_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
