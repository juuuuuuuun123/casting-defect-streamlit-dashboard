from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .config import PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs
from .metrics import classification_metrics
from .run_feature_experiments import build_feature_matrix, clip_outliers, make_estimator


def feature_names(image_size: int = 32) -> list[str]:
    names = [f"pixel_r{row:02d}_c{col:02d}" for row in range(image_size) for col in range(image_size)]
    return names + ["width", "height", "brightness", "contrast"]


def cohen_d(left: pd.Series, right: pd.Series) -> float:
    left = left.dropna().astype(float)
    right = right.dropna().astype(float)
    pooled = np.sqrt(((left.std(ddof=1) ** 2) + (right.std(ddof=1) ** 2)) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((left.mean() - right.mean()) / pooled)


def group_difference(frame: pd.DataFrame, column: str) -> dict:
    defect = frame.loc[frame["label"] == 1, column]
    ok = frame.loc[frame["label"] == 0, column]
    ttest = stats.ttest_ind(defect, ok, equal_var=False, nan_policy="omit")
    mann = stats.mannwhitneyu(defect.dropna(), ok.dropna(), alternative="two-sided")
    return {
        "feature": column,
        "defect_mean": float(defect.mean()),
        "ok_mean": float(ok.mean()),
        "mean_diff_defect_minus_ok": float(defect.mean() - ok.mean()),
        "cohen_d": cohen_d(defect, ok),
        "ttest_pvalue": float(ttest.pvalue),
        "mannwhitney_pvalue": float(mann.pvalue),
    }


def summarize_error_groups(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame()
    rows = []
    for group_name, group in errors.groupby("error_margin_group"):
        rows.append(
            {
                "error_margin_group": group_name,
                "n": int(len(group)),
                "false_negative": int((group["error_type"] == "False Negative").sum()),
                "false_positive": int((group["error_type"] == "False Positive").sum()),
                "mean_prob_defect": float(group["prob_defect"].mean()),
                "mean_margin": float(group["margin_from_threshold"].mean()),
                "mean_brightness": float(group["brightness"].mean()),
                "mean_contrast": float(group["contrast"].mean()),
                "mean_top_region_intensity": float(group["top_region_intensity"].mean()),
                "mean_low_region_intensity": float(group["low_region_intensity"].mean()),
            }
        )
    return pd.DataFrame(rows)


def generate_dashboard_insights(image_size: int = 32) -> None:
    ensure_project_dirs()
    metadata = pd.read_csv(PROCESSED_DIR / "metadata.csv")
    experiments = pd.read_csv(REPORTS_DIR / "experiments.csv")
    holdout = pd.read_csv(REPORTS_DIR / "holdout_predictions.csv")
    summary = json.loads((REPORTS_DIR / "holdout_summary.json").read_text(encoding="utf-8"))
    best_experiment = summary["best_experiment"]

    x, y, paths = build_feature_matrix(metadata, image_size=image_size)
    path_to_index = {path: idx for idx, path in enumerate(paths)}
    train_valid = metadata[metadata["split"] == "train_valid"]
    holdout_meta = metadata[metadata["split"] == "holdout"].copy().reset_index(drop=True)
    train_idx = [path_to_index[path] for path in train_valid["relative_path"]]
    holdout_idx = [path_to_index[path] for path in holdout_meta["relative_path"]]
    x_train, x_holdout = x[train_idx], x[holdout_idx]
    y_train = y[train_idx]
    if best_experiment in {"F3_lightgbm_weighted", "F4_xgboost_weighted"}:
        x_train, x_holdout = clip_outliers(x_train, x_holdout)

    estimator = make_estimator(best_experiment, y_train)
    estimator.fit(x_train, y_train)
    model = estimator.named_steps["model"]
    names = feature_names(image_size=image_size)
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(names), dtype=float)

    importance = pd.DataFrame({"feature": names, "importance": importances})
    importance["importance_share"] = importance["importance"] / max(float(importance["importance"].sum()), 1.0)
    importance["feature_type"] = np.where(importance["feature"].str.startswith("pixel_"), "pixel", "metadata")
    importance.sort_values("importance", ascending=False).to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    pixel_importance = importances[: image_size * image_size]
    top_n = max(10, int(len(pixel_importance) * 0.05))
    top_pixel_idx = np.argsort(pixel_importance)[-top_n:]
    low_pixel_idx = np.argsort(pixel_importance)[:top_n]

    enriched = metadata.copy()
    matrix = x[:, : image_size * image_size]
    enriched["top_region_intensity"] = matrix[:, top_pixel_idx].mean(axis=1)
    enriched["low_region_intensity"] = matrix[:, low_pixel_idx].mean(axis=1)
    enriched["label_name"] = enriched["label"].map({0: "OK", 1: "Defective"})

    stat_features = ["brightness", "contrast", "top_region_intensity", "low_region_intensity"]
    group_stats = pd.DataFrame([group_difference(enriched, column) for column in stat_features])
    group_stats.to_csv(REPORTS_DIR / "feature_group_statistics.csv", index=False)

    holdout_enriched = holdout.merge(
        enriched[["relative_path", "top_region_intensity", "low_region_intensity", "brightness", "contrast", "path"]],
        on="relative_path",
        how="left",
        suffixes=("", "_meta"),
    )
    holdout_enriched["margin_from_threshold"] = (holdout_enriched["prob_defect"] - 0.5).abs()
    holdout_enriched["error_type"] = np.select(
        [
            (holdout_enriched["label"] == 1) & (holdout_enriched["prediction"] == 0),
            (holdout_enriched["label"] == 0) & (holdout_enriched["prediction"] == 1),
        ],
        ["False Negative", "False Positive"],
        default="Correct",
    )
    errors = holdout_enriched[~holdout_enriched["is_correct"]].copy()
    errors["error_margin_group"] = np.where(errors["margin_from_threshold"] <= 0.15, "near_threshold", "high_confidence")
    errors["label_issue_risk"] = np.where(
        errors["margin_from_threshold"] >= 0.4,
        "review_label_or_outlier",
        "likely_borderline_or_preprocessing_sensitive",
    )
    errors.sort_values(["error_margin_group", "margin_from_threshold"], ascending=[True, False]).to_csv(
        REPORTS_DIR / "error_cases_enriched.csv",
        index=False,
    )
    summarize_error_groups(errors).to_csv(REPORTS_DIR / "error_group_summary.csv", index=False)

    corr_columns = ["label", "prob_defect", "brightness", "contrast", "top_region_intensity", "low_region_intensity"]
    correlations = holdout_enriched[corr_columns].corr(numeric_only=True).reset_index().rename(columns={"index": "variable"})
    correlations.to_csv(REPORTS_DIR / "feature_correlations.csv", index=False)

    cv_summary = (
        experiments.groupby("experiment")[["f1", "recall", "pr_auc", "false_negative_rate"]]
        .agg(["mean", "std"])
        .round(5)
    )
    best_cv = cv_summary.loc[best_experiment]
    insights = {
        "best_experiment": best_experiment,
        "holdout_metrics": summary["metrics"],
        "best_cv": {
            "f1_mean": float(best_cv[("f1", "mean")]),
            "recall_mean": float(best_cv[("recall", "mean")]),
            "pr_auc_mean": float(best_cv[("pr_auc", "mean")]),
            "false_negative_rate_mean": float(best_cv[("false_negative_rate", "mean")]),
        },
        "top_feature_summary": {
            "top_pixel_count": int(top_n),
            "metadata_importance_share": float(importance.loc[importance["feature_type"] == "metadata", "importance_share"].sum()),
            "pixel_importance_share": float(importance.loc[importance["feature_type"] == "pixel", "importance_share"].sum()),
        },
        "error_summary": {
            "total_errors": int(len(errors)),
            "false_negatives": int((errors["error_type"] == "False Negative").sum()),
            "false_positives": int((errors["error_type"] == "False Positive").sum()),
            "high_confidence_errors": int((errors["error_margin_group"] == "high_confidence").sum()),
            "near_threshold_errors": int((errors["error_margin_group"] == "near_threshold").sum()),
        },
        "recommendations": [
            "Inspect high-confidence errors first; these are the strongest candidates for label noise, duplicate leakage, or systematic preprocessing artifacts.",
            "Compare threshold 0.5 with a recall-oriented operating point because false negatives are operationally more expensive in defect inspection.",
            "Move from low-resolution pixel features to transfer learning CNN or GradCAM to verify that the model focuses on defect regions rather than borders or lighting.",
            "Use hard-example mining by adding repeated augmentation for high-confidence false negatives and borderline near-threshold cases.",
        ],
    }
    (REPORTS_DIR / "dashboard_insights.json").write_text(json.dumps(insights, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(insights, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    generate_dashboard_insights()
