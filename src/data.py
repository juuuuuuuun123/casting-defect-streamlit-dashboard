from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageStat, UnidentifiedImageError
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import PROCESSED_DIR, RAW_DIR, ExperimentConfig, ensure_project_dirs


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def discover_dataset_root(raw_root: Path = RAW_DIR) -> Path:
    """Prefer the original train/test image tree over the 512x512 sample folder."""
    preferred = raw_root / "casting_data" / "casting_data"
    if preferred.exists():
        return preferred
    return raw_root


def list_image_files(root: Path = RAW_DIR) -> list[Path]:
    root = discover_dataset_root(root)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def infer_label(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    if any(part in {"def_front", "defective", "def", "bad"} for part in parts):
        return "def_front"
    if any(part in {"ok_front", "ok", "normal", "good"} for part in parts):
        return "ok_front"
    return None


def file_sha1(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_quality_record(path: Path, raw_root: Path = RAW_DIR) -> dict:
    label = infer_label(path)
    record = {
        "path": str(path),
        "relative_path": str(path.relative_to(raw_root)),
        "label_name": label,
        "label": 1 if label == "def_front" else 0 if label == "ok_front" else np.nan,
        "width": np.nan,
        "height": np.nan,
        "channels": np.nan,
        "brightness": np.nan,
        "contrast": np.nan,
        "sha1": "",
        "readable": False,
        "issue": "",
    }
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            stat = ImageStat.Stat(image)
            record.update(
                {
                    "width": image.width,
                    "height": image.height,
                    "channels": len(image.getbands()),
                    "brightness": float(np.mean(stat.mean)),
                    "contrast": float(np.mean(stat.stddev)),
                    "sha1": file_sha1(path),
                    "readable": True,
                }
            )
    except (OSError, UnidentifiedImageError) as exc:
        record["issue"] = f"{type(exc).__name__}: {exc}"
    return record


def build_metadata(raw_root: Path = RAW_DIR, config: ExperimentConfig | None = None) -> pd.DataFrame:
    ensure_project_dirs()
    raw_root = discover_dataset_root(raw_root)
    config = config or ExperimentConfig()
    records = [image_quality_record(path, raw_root) for path in list_image_files(raw_root)]
    metadata = pd.DataFrame(records)

    if metadata.empty:
        raise FileNotFoundError(f"No image files found under {raw_root}")

    issues = metadata[(~metadata["readable"]) | metadata["label"].isna()].copy()
    if not issues.empty:
        issues.to_csv(PROCESSED_DIR.parent / "processed" / "data_quality_issues.csv", index=False)

    usable = metadata[metadata["readable"] & metadata["label"].notna()].copy()
    usable["label"] = usable["label"].astype(int)
    usable["duplicate_sha1"] = usable.duplicated("sha1", keep=False)

    train_valid, holdout = train_test_split(
        usable,
        test_size=config.holdout_size,
        random_state=config.seed,
        stratify=usable["label"],
    )
    usable["split"] = "train_valid"
    usable.loc[holdout.index, "split"] = "holdout"
    usable["fold"] = -1

    cv_pool = usable[usable["split"] == "train_valid"].copy()
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    for fold, (_, valid_idx) in enumerate(skf.split(cv_pool, cv_pool["label"])):
        valid_indices = cv_pool.iloc[valid_idx].index
        usable.loc[valid_indices, "fold"] = fold

    output_path = PROCESSED_DIR / "metadata.csv"
    usable.to_csv(output_path, index=False)
    return usable


def summarize_metadata(metadata: pd.DataFrame) -> dict:
    duplicate_count = int(metadata["duplicate_sha1"].sum()) if "duplicate_sha1" in metadata else 0
    return {
        "n_images": int(len(metadata)),
        "n_defective": int((metadata["label"] == 1).sum()),
        "n_ok": int((metadata["label"] == 0).sum()),
        "holdout_images": int((metadata["split"] == "holdout").sum()),
        "train_valid_images": int((metadata["split"] == "train_valid").sum()),
        "duplicate_sha1_count": duplicate_count,
        "width_min": int(metadata["width"].min()),
        "width_max": int(metadata["width"].max()),
        "height_min": int(metadata["height"].min()),
        "height_max": int(metadata["height"].max()),
        "brightness_mean": float(metadata["brightness"].mean()),
        "contrast_mean": float(metadata["contrast"].mean()),
    }


def save_summary(metadata: pd.DataFrame, path: Path | None = None) -> dict:
    path = path or PROCESSED_DIR / "data_summary.json"
    summary = summarize_metadata(metadata)
    pd.Series(summary).to_json(path, force_ascii=False, indent=2)
    return summary


def iter_fold_rows(metadata: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv_pool = metadata[metadata["split"] == "train_valid"].copy()
    train_df = cv_pool[cv_pool["fold"] != fold].copy()
    valid_df = cv_pool[cv_pool["fold"] == fold].copy()
    return train_df, valid_df
