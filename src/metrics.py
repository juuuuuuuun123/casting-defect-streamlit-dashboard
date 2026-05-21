from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.stats.contingency_tables import mcnemar


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_threshold_for_recall(y_true, y_prob, min_recall: float = 0.95) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    candidates = []
    for threshold in thresholds:
        metrics = classification_metrics(y_true, y_prob, threshold)
        if metrics["recall"] >= min_recall:
            candidates.append((threshold, metrics["f1"], metrics["false_negative_rate"]))
    if not candidates:
        return 0.5
    return float(max(candidates, key=lambda row: row[1])[0])


def paired_metric_tests(experiments: pd.DataFrame, metric: str, baseline: str, challenger: str) -> dict:
    left = experiments[experiments["experiment"] == baseline][["fold", metric]].rename(columns={metric: "baseline"})
    right = experiments[experiments["experiment"] == challenger][["fold", metric]].rename(columns={metric: "challenger"})
    paired = left.merge(right, on="fold").dropna()
    if len(paired) < 2:
        return {"metric": metric, "baseline": baseline, "challenger": challenger, "n": len(paired)}
    diff = paired["challenger"] - paired["baseline"]
    ttest = stats.ttest_rel(paired["challenger"], paired["baseline"])
    wilcoxon = stats.wilcoxon(paired["challenger"], paired["baseline"], zero_method="zsplit")
    return {
        "metric": metric,
        "baseline": baseline,
        "challenger": challenger,
        "n": int(len(paired)),
        "mean_diff": float(diff.mean()),
        "ttest_pvalue": float(ttest.pvalue),
        "wilcoxon_pvalue": float(wilcoxon.pvalue),
    }


def mcnemar_test(y_true, baseline_prob, challenger_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    base_correct = ((np.asarray(baseline_prob) >= threshold).astype(int) == y_true)
    chal_correct = ((np.asarray(challenger_prob) >= threshold).astype(int) == y_true)
    table = np.array(
        [
            [np.sum(base_correct & chal_correct), np.sum(base_correct & ~chal_correct)],
            [np.sum(~base_correct & chal_correct), np.sum(~base_correct & ~chal_correct)],
        ]
    )
    result = mcnemar(table, exact=False, correction=True)
    return {
        "table_00_both_correct": int(table[0, 0]),
        "table_01_baseline_only": int(table[0, 1]),
        "table_10_challenger_only": int(table[1, 0]),
        "table_11_both_wrong": int(table[1, 1]),
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def bootstrap_ci(y_true, y_prob, metric_name: str = "f1", threshold: float = 0.5, n_bootstrap: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(classification_metrics(y_true[idx], y_prob[idx], threshold)[metric_name])
    return {
        "metric": metric_name,
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "n_bootstrap": int(len(values)),
    }
