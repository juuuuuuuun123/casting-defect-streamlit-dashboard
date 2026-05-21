from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, ExperimentConfig, ensure_project_dirs
from .data import build_metadata, iter_fold_rows, save_summary
from .metrics import classification_metrics, find_threshold_for_recall, paired_metric_tests
from .models import build_feature_extractor, build_model
from .torch_data import CastingImageDataset, build_transforms, make_weighted_sampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        loss = self.alpha * (1 - pt).pow(self.gamma) * bce
        return loss.mean()


def make_loss(train_df: pd.DataFrame, loss_name: str) -> nn.Module:
    if loss_name == "focal":
        return FocalLoss()
    if loss_name == "weighted_bce":
        positives = float((train_df["label"] == 1).sum())
        negatives = float((train_df["label"] == 0).sum())
        pos_weight = torch.tensor([negatives / max(positives, 1.0)])
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    losses = []
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        criterion = criterion.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for images, labels, paths in loader:
        logits = model(images.to(device))
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        for path, label, prob in zip(paths, labels.numpy(), probs):
            rows.append({"relative_path": path, "label": int(label), "prob_defect": float(prob)})
    return pd.DataFrame(rows)


def train_fold(
    experiment: dict,
    fold: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    train_tf = build_transforms(config.image_size, train=True, augmentation=experiment.get("augmentation", "weak"))
    valid_tf = build_transforms(config.image_size, train=False)
    train_ds = CastingImageDataset(train_df, train_tf)
    valid_ds = CastingImageDataset(valid_df, valid_tf)

    sampler = make_weighted_sampler(train_df) if experiment.get("sampler") == "weighted" else None
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = build_model(
        experiment["model_name"],
        pretrained=experiment.get("pretrained", True),
        freeze_backbone=experiment.get("freeze_backbone", False),
    ).to(device)

    criterion = make_loss(train_df, experiment.get("loss", "bce"))
    optimizer_name = experiment.get("optimizer", "adamw").lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_f1 = -1.0
    best_state = None
    patience = 0
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_pred = predict(model, valid_loader, device)
        valid_metrics = classification_metrics(valid_pred["label"], valid_pred["prob_defect"])
        if valid_metrics["f1"] > best_f1:
            best_f1 = valid_metrics["f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        print(
            f"{experiment['experiment']} fold={fold} epoch={epoch + 1} "
            f"loss={train_loss:.4f} f1={valid_metrics['f1']:.4f} recall={valid_metrics['recall']:.4f}"
        )
        if patience >= experiment.get("early_stopping_patience", 5):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    fold_pred = predict(model, valid_loader, device)
    threshold = find_threshold_for_recall(fold_pred["label"], fold_pred["prob_defect"], min_recall=0.95)
    metrics = classification_metrics(fold_pred["label"], fold_pred["prob_defect"], threshold=threshold)
    metrics.update({"experiment": experiment["experiment"], "fold": fold, "threshold_recall95": threshold})

    model_path = MODELS_DIR / f"{experiment['experiment']}_fold{fold}.pt"
    torch.save({"model_state": model.state_dict(), "experiment": experiment, "metrics": metrics}, model_path)
    fold_pred["experiment"] = experiment["experiment"]
    fold_pred["fold"] = fold
    return metrics, fold_pred


@torch.no_grad()
def extract_embeddings(frame: pd.DataFrame, config: ExperimentConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    transform = build_transforms(config.image_size, train=False)
    dataset = CastingImageDataset(frame, transform)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    extractor = build_feature_extractor("resnet18").to(device)
    vectors, labels, paths = [], [], []
    for images, batch_labels, batch_paths in tqdm(loader, desc="extract embeddings"):
        batch_vectors = extractor(images.to(device)).detach().cpu().numpy()
        vectors.append(batch_vectors)
        labels.extend(batch_labels.numpy().astype(int).tolist())
        paths.extend(list(batch_paths))
    return np.vstack(vectors), np.asarray(labels), paths


def run_embedding_ml(metadata: pd.DataFrame, config: ExperimentConfig, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, predictions = [], []
    for fold in range(config.num_folds):
        train_df, valid_df = iter_fold_rows(metadata, fold)
        x_train, y_train, _ = extract_embeddings(train_df, config, device)
        x_valid, y_valid, valid_paths = extract_embeddings(valid_df, config, device)
        models = {
            "embed_logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "embed_random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=config.seed),
        }
        try:
            from lightgbm import LGBMClassifier

            models["embed_lightgbm"] = LGBMClassifier(class_weight="balanced", random_state=config.seed, verbose=-1)
        except Exception:
            pass
        try:
            from xgboost import XGBClassifier

            scale_pos_weight = max((y_train == 0).sum(), 1) / max((y_train == 1).sum(), 1)
            models["embed_xgboost"] = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=config.seed,
            )
        except Exception:
            pass

        for name, estimator in models.items():
            estimator.fit(x_train, y_train)
            probs = estimator.predict_proba(x_valid)[:, 1]
            threshold = find_threshold_for_recall(y_valid, probs, min_recall=0.95)
            metrics = classification_metrics(y_valid, probs, threshold=threshold)
            metrics.update({"experiment": name, "fold": fold, "threshold_recall95": threshold})
            rows.append(metrics)
            predictions.extend(
                {
                    "relative_path": path,
                    "label": int(label),
                    "prob_defect": float(prob),
                    "experiment": name,
                    "fold": fold,
                }
                for path, label, prob in zip(valid_paths, y_valid, probs)
            )
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def experiment_grid() -> list[dict]:
    return [
        {"experiment": "B1_small_cnn_basic", "model_name": "small_cnn", "loss": "bce", "augmentation": "none"},
        {"experiment": "B2_small_cnn_weighted", "model_name": "small_cnn", "loss": "weighted_bce", "augmentation": "weak"},
        {"experiment": "T1_resnet18_weighted", "model_name": "resnet18", "loss": "weighted_bce", "augmentation": "weak"},
        {"experiment": "T2_mobilenet_v3_weighted", "model_name": "mobilenet_v3_small", "loss": "weighted_bce", "augmentation": "weak"},
        {"experiment": "T3_efficientnet_focal", "model_name": "efficientnet_b0", "loss": "focal", "augmentation": "medium"},
        {"experiment": "A1_resnet18_sampler", "model_name": "resnet18", "loss": "bce", "augmentation": "weak", "sampler": "weighted"},
    ]


def run_cv(config: ExperimentConfig) -> None:
    ensure_project_dirs()
    set_seed(config.seed)
    device = get_device()
    metadata_path = PROCESSED_DIR / "metadata.csv"
    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else build_metadata(config=config)
    save_summary(metadata)

    metric_rows, pred_frames = [], []
    for experiment in experiment_grid():
        for fold in range(config.num_folds):
            train_df, valid_df = iter_fold_rows(metadata, fold)
            metrics, fold_pred = train_fold(experiment, fold, train_df, valid_df, config, device)
            metric_rows.append(metrics)
            pred_frames.append(fold_pred)

    ml_metrics, ml_preds = run_embedding_ml(metadata, config, device)
    metrics_df = pd.concat([pd.DataFrame(metric_rows), ml_metrics], ignore_index=True)
    oof_df = pd.concat(pred_frames + [ml_preds], ignore_index=True)

    metrics_df.to_csv(REPORTS_DIR / "experiments.csv", index=False)
    oof_df.to_csv(REPORTS_DIR / "oof_predictions.csv", index=False)

    stats_rows = []
    for challenger in sorted(set(metrics_df["experiment"]) - {"B1_small_cnn_basic"}):
        for metric in ["f1", "recall", "pr_auc", "false_negative_rate"]:
            stats_rows.append(paired_metric_tests(metrics_df, metric, "B1_small_cnn_basic", challenger))
    pd.DataFrame(stats_rows).to_csv(REPORTS_DIR / "statistical_tests.csv", index=False)

    with (REPORTS_DIR / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config.__dict__, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5-fold CV experiments for casting defect classification.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cv(
        ExperimentConfig(
            seed=args.seed,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
    )
