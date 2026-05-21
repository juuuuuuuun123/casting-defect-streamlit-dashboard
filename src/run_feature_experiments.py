from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PROCESSED_DIR, REPORTS_DIR, ExperimentConfig, ensure_project_dirs
from .data import build_metadata, iter_fold_rows, save_summary
from .metrics import (
    bootstrap_ci,
    classification_metrics,
    find_threshold_for_recall,
    mcnemar_test,
    paired_metric_tests,
)


def image_vector(path: str, image_size: int = 32) -> np.ndarray:
    image = Image.open(path).convert("L").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def build_feature_matrix(metadata: pd.DataFrame, image_size: int = 32) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cache_path = PROCESSED_DIR / f"image_features_{image_size}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["x"], cached["y"], cached["paths"].tolist()

    vectors = []
    for path in metadata["path"]:
        vectors.append(image_vector(path, image_size=image_size))
    pixel_features = np.vstack(vectors)
    numeric_features = metadata[["width", "height", "brightness", "contrast"]].to_numpy(dtype=np.float32)
    x = np.hstack([pixel_features, numeric_features])
    y = metadata["label"].to_numpy(dtype=int)
    paths = metadata["relative_path"].tolist()
    np.savez_compressed(cache_path, x=x, y=y, paths=np.asarray(paths, dtype=object))
    return x, y, paths


def make_estimator(name: str, y_train: np.ndarray):
    if name == "F0_logreg_basic":
        model = LogisticRegression(max_iter=1000)
    elif name == "F1_logreg_class_weight":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif name == "F2_random_forest_balanced":
        model = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1)
    elif name == "F3_lightgbm_weighted":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise RuntimeError("lightgbm is required for F3_lightgbm_weighted") from exc
        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
    elif name == "F4_xgboost_weighted":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError("xgboost is required for F4_xgboost_weighted") from exc
        scale_pos_weight = max((y_train == 0).sum(), 1) / max((y_train == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown estimator: {name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=name.startswith("F0") or name.startswith("F1"))),
            ("model", model),
        ]
    )


def clip_outliers(x_train: np.ndarray, x_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.nanpercentile(x_train, 1, axis=0)
    high = np.nanpercentile(x_train, 99, axis=0)
    return np.clip(x_train, low, high), np.clip(x_valid, low, high)


def run_feature_experiments(image_size: int = 32) -> None:
    ensure_project_dirs()
    metadata_path = PROCESSED_DIR / "metadata.csv"
    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else build_metadata(config=ExperimentConfig())
    save_summary(metadata)

    x, y, paths = build_feature_matrix(metadata, image_size=image_size)
    path_to_index = {path: idx for idx, path in enumerate(paths)}

    experiment_names = [
        "F0_logreg_basic",
        "F1_logreg_class_weight",
        "F2_random_forest_balanced",
        "F3_lightgbm_weighted",
        "F4_xgboost_weighted",
    ]
    metrics_rows = []
    oof_rows = []

    for experiment in experiment_names:
        for fold in range(5):
            train_df, valid_df = iter_fold_rows(metadata, fold)
            train_idx = [path_to_index[path] for path in train_df["relative_path"]]
            valid_idx = [path_to_index[path] for path in valid_df["relative_path"]]
            x_train, x_valid = x[train_idx], x[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            if experiment in {"F3_lightgbm_weighted", "F4_xgboost_weighted"}:
                x_train, x_valid = clip_outliers(x_train, x_valid)

            estimator = make_estimator(experiment, y_train)
            estimator.fit(x_train, y_train)
            probs = estimator.predict_proba(x_valid)[:, 1]
            threshold = find_threshold_for_recall(y_valid, probs, min_recall=0.95)
            metrics = classification_metrics(y_valid, probs, threshold=threshold)
            metrics.update({"experiment": experiment, "fold": fold, "threshold_recall95": threshold})
            metrics_rows.append(metrics)
            oof_rows.extend(
                {
                    "relative_path": path,
                    "label": int(label),
                    "prob_defect": float(prob),
                    "experiment": experiment,
                    "fold": fold,
                }
                for path, label, prob in zip(valid_df["relative_path"], y_valid, probs)
            )

    experiments = pd.DataFrame(metrics_rows)
    oof = pd.DataFrame(oof_rows)
    experiments.to_csv(REPORTS_DIR / "experiments.csv", index=False)
    oof.to_csv(REPORTS_DIR / "oof_predictions.csv", index=False)

    stats_rows = []
    for challenger in sorted(set(experiments["experiment"]) - {"F0_logreg_basic"}):
        for metric in ["f1", "recall", "roc_auc", "pr_auc", "false_negative_rate"]:
            stats_rows.append(paired_metric_tests(experiments, metric, "F0_logreg_basic", challenger))
    pd.DataFrame(stats_rows).to_csv(REPORTS_DIR / "statistical_tests.csv", index=False)

    best_experiment = experiments.groupby("experiment")["f1"].mean().sort_values(ascending=False).index[0]
    train_valid = metadata[metadata["split"] == "train_valid"]
    holdout = metadata[metadata["split"] == "holdout"].copy().reset_index(drop=True)
    train_idx = [path_to_index[path] for path in train_valid["relative_path"]]
    holdout_idx = [path_to_index[path] for path in holdout["relative_path"]]
    x_train, x_holdout = x[train_idx], x[holdout_idx]
    y_train, y_holdout = y[train_idx], y[holdout_idx]
    if best_experiment in {"F3_lightgbm_weighted", "F4_xgboost_weighted"}:
        x_train, x_holdout = clip_outliers(x_train, x_holdout)
    best_estimator = make_estimator(best_experiment, y_train)
    best_estimator.fit(x_train, y_train)
    holdout["prob_defect"] = best_estimator.predict_proba(x_holdout)[:, 1]
    holdout["prediction"] = (holdout["prob_defect"] >= 0.5).astype(int)
    holdout["is_correct"] = holdout["prediction"] == holdout["label"]
    holdout["experiment"] = best_experiment
    holdout.to_csv(REPORTS_DIR / "holdout_predictions.csv", index=False)

    holdout_metrics = classification_metrics(y_holdout, holdout["prob_defect"])
    holdout_summary = {
        "best_experiment": best_experiment,
        "metrics": holdout_metrics,
        "bootstrap_ci": [
            bootstrap_ci(y_holdout, holdout["prob_defect"], metric_name=metric, n_bootstrap=1000)
            for metric in ["recall", "f1", "roc_auc", "pr_auc"]
        ],
    }
    (REPORTS_DIR / "holdout_summary.json").write_text(json.dumps(holdout_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline = oof[oof["experiment"] == "F0_logreg_basic"].sort_values("relative_path")
    best = oof[oof["experiment"] == best_experiment].sort_values("relative_path")
    if best_experiment != "F0_logreg_basic" and len(baseline) == len(best):
        test = mcnemar_test(best["label"], baseline["prob_defect"], best["prob_defect"])
        (REPORTS_DIR / "mcnemar_best_vs_baseline.json").write_text(
            json.dumps(test, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (REPORTS_DIR / "feature_experiment_config.json").write_text(
        json.dumps({"image_size": image_size, "note": "Fast 5-fold feature experiments for dashboard artifacts."}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_feature_experiments()
