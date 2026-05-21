from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, ExperimentConfig
from .metrics import bootstrap_ci, classification_metrics, mcnemar_test
from .models import build_model
from .torch_data import CastingImageDataset, build_transforms


@torch.no_grad()
def predict_checkpoint(checkpoint_path: Path, holdout_df: pd.DataFrame, config: ExperimentConfig, device: torch.device) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    experiment = checkpoint["experiment"]
    model = build_model(
        experiment["model_name"],
        pretrained=False,
        freeze_backbone=experiment.get("freeze_backbone", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(
        CastingImageDataset(holdout_df, build_transforms(config.image_size, train=False)),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    probs = []
    for images, _, _ in loader:
        logits = model(images.to(device))
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    return np.asarray(probs)


def evaluate_holdout() -> None:
    metadata = pd.read_csv(PROCESSED_DIR / "metadata.csv")
    metrics = pd.read_csv(REPORTS_DIR / "experiments.csv")
    config_path = REPORTS_DIR / "run_config.json"
    config = ExperimentConfig(**json.loads(config_path.read_text(encoding="utf-8"))) if config_path.exists() else ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_experiment = (
        metrics.groupby("experiment")["f1"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    holdout_df = metadata[metadata["split"] == "holdout"].copy().reset_index(drop=True)
    fold_probs = []
    for fold in sorted(metrics[metrics["experiment"] == best_experiment]["fold"].unique()):
        checkpoint_path = MODELS_DIR / f"{best_experiment}_fold{int(fold)}.pt"
        if checkpoint_path.exists():
            fold_probs.append(predict_checkpoint(checkpoint_path, holdout_df, config, device))
    if not fold_probs:
        raise FileNotFoundError(f"No fold checkpoints found for {best_experiment}")

    holdout_df["prob_defect"] = np.mean(np.vstack(fold_probs), axis=0)
    holdout_df["prediction"] = (holdout_df["prob_defect"] >= 0.5).astype(int)
    holdout_df["is_correct"] = holdout_df["prediction"] == holdout_df["label"]
    holdout_df["experiment"] = best_experiment
    holdout_df.to_csv(REPORTS_DIR / "holdout_predictions.csv", index=False)

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity", "false_negative_rate"]
    holdout_metrics = classification_metrics(holdout_df["label"], holdout_df["prob_defect"])
    ci_rows = [
        bootstrap_ci(holdout_df["label"], holdout_df["prob_defect"], metric_name=name, n_bootstrap=1000)
        for name in ["recall", "f1", "roc_auc", "pr_auc"]
    ]
    output = {
        "best_experiment": best_experiment,
        "metrics": {key: holdout_metrics[key] for key in metric_names},
        "bootstrap_ci": ci_rows,
    }
    (REPORTS_DIR / "holdout_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    oof = pd.read_csv(REPORTS_DIR / "oof_predictions.csv")
    baseline = oof[oof["experiment"] == "B1_small_cnn_basic"].sort_values("relative_path")
    best = oof[oof["experiment"] == best_experiment].sort_values("relative_path")
    if best_experiment != "B1_small_cnn_basic" and len(baseline) == len(best):
        test = mcnemar_test(best["label"], baseline["prob_defect"], best["prob_defect"])
        (REPORTS_DIR / "mcnemar_best_vs_baseline.json").write_text(
            json.dumps(test, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    evaluate_holdout()
