from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve


def ko(text: str) -> str:
    """Render Korean text while keeping this source file ASCII-safe."""
    return text.encode("ascii").decode("unicode_escape")


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"


st.set_page_config(page_title="Casting Defect Analysis Dashboard", layout="wide")
st.title(ko(r"\uc8fc\uc870\ud488 \ubd88\ub7c9 \uac80\uc0ac AI \ub300\uc2dc\ubcf4\ub4dc"))
st.caption(ko(r"\uc2e4\ud5d8 \uacb0\uacfc, \uc785\ub825 \ubcc0\uc218 \ubd84\uc11d, \uc624\ubd84\ub958 \uc2ec\ud654 \ubd84\uc11d\uc744 \ud1b5\ud574 \ub2e4\uc74c \ubaa8\ub378 \uace0\ub3c4\ud654 \ubc29\ud5a5\uc744 \ub3c4\ucd9c\ud569\ub2c8\ub2e4."))


@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def flatten_cv_summary(experiments: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_negative_rate"]
    summary = experiments.groupby("experiment")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values("f1_mean", ascending=False)


def pixel_importance_grid(importance: pd.DataFrame, image_size: int = 32) -> np.ndarray:
    grid = np.zeros((image_size, image_size), dtype=float)
    for _, row in importance[importance["feature"].str.startswith("pixel_")].iterrows():
        match = re.match(r"pixel_r(\d+)_c(\d+)", row["feature"])
        if match:
            grid[int(match.group(1)), int(match.group(2))] = float(row["importance"])
    return grid


def format_pvalue(value: float) -> str:
    if pd.isna(value):
        return ""
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def resolve_image_path(row: pd.Series) -> Path | None:
    """Return a displayable local image path when raw images exist locally."""
    for column in ["path", "path_meta"]:
        value = row.get(column)
        if isinstance(value, str) and value:
            path = Path(value)
            if path.exists():
                return path
    relative_path = row.get("relative_path")
    if isinstance(relative_path, str) and relative_path:
        candidate = ROOT / "data" / "raw" / "casting_data" / "casting_data" / Path(relative_path)
        if candidate.exists():
            return candidate
    return None


metadata = read_csv(PROCESSED / "metadata.csv")
experiments = read_csv(REPORTS / "experiments.csv")
stats_tests = read_csv(REPORTS / "statistical_tests.csv")
oof = read_csv(REPORTS / "oof_predictions.csv")
holdout = read_csv(REPORTS / "holdout_predictions.csv")
importance = read_csv(REPORTS / "feature_importance.csv")
feature_stats = read_csv(REPORTS / "feature_group_statistics.csv")
correlations = read_csv(REPORTS / "feature_correlations.csv")
errors = read_csv(REPORTS / "error_cases_enriched.csv")
error_groups = read_csv(REPORTS / "error_group_summary.csv")
holdout_summary = read_json(REPORTS / "holdout_summary.json")
dashboard_insights = read_json(REPORTS / "dashboard_insights.json")

if metadata.empty:
    st.warning(ko(r"`metadata.csv`\uac00 \uc5c6\uc2b5\ub2c8\ub2e4. `python -m src.prepare_data`\ub97c \uc2e4\ud589\ud558\uc138\uc694."))
    st.stop()

tab_experiments, tab_features, tab_errors = st.tabs(
    [
        ko(r"\ud0ed1: \uc2e4\ud5d8 \uacb0\uacfc"),
        ko(r"\ud0ed2: \uc785\ub825 \ubcc0\uc218 \ubd84\uc11d"),
        ko(r"\ud0ed3: \uc624\ubd84\ub958 \ub370\uc774\ud130 \uc2ec\ud654 \ubd84\uc11d"),
    ]
)

with tab_experiments:
    st.header(ko(r"\uc2e4\ud5d8 \uacb0\uacfc\uc640 \ud1b5\uacc4\uac80\uc99d"))
    if experiments.empty:
        st.info(ko(r"\uc2e4\ud5d8 \uacb0\uacfc\uac00 \uc5c6\uc2b5\ub2c8\ub2e4. `python -m src.run_feature_experiments`\ub97c \uc2e4\ud589\ud558\uc138\uc694."))
    else:
        cv_summary = flatten_cv_summary(experiments)
        best_model = dashboard_insights.get("best_experiment", cv_summary.iloc[0]["experiment"])
        metrics = holdout_summary.get("metrics", {})

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(ko(r"\ucd5c\uace0 \ubaa8\ub378"), best_model)
        c2.metric("Holdout F1", f"{metrics.get('f1', np.nan):.4f}")
        c3.metric("Holdout AUROC", f"{metrics.get('roc_auc', np.nan):.4f}")
        c4.metric("Holdout Recall", f"{metrics.get('recall', np.nan):.4f}")
        c5.metric(ko(r"\ubd88\ub7c9 \ubbf8\uac80\ucd9c\ub960"), f"{metrics.get('false_negative_rate', np.nan):.4f}")

        display_cols = [
            "experiment",
            "f1_mean",
            "f1_std",
            "recall_mean",
            "recall_std",
            "roc_auc_mean",
            "roc_auc_std",
            "pr_auc_mean",
            "pr_auc_std",
            "false_negative_rate_mean",
            "false_negative_rate_std",
        ]
        st.subheader(ko(r"\uc804\uccb4 \uc2e4\ud5d8 \ud14c\uc774\ube14"))
        st.dataframe(cv_summary[display_cols].round(5), width="stretch")

        st.subheader(ko(r"Baseline \ub300\ube44 \ud1b5\uacc4\uac80\uc99d"))
        if stats_tests.empty:
            st.info(ko(r"\ud1b5\uacc4\uac80\uc99d \ud30c\uc77c\uc774 \uc5c6\uc2b5\ub2c8\ub2e4."))
        else:
            stats_display = stats_tests.copy()
            stats_display["ttest_pvalue"] = stats_display["ttest_pvalue"].map(format_pvalue)
            stats_display["wilcoxon_pvalue"] = stats_display["wilcoxon_pvalue"].map(format_pvalue)
            st.dataframe(stats_display, width="stretch")
            st.caption(ko(r"paired t-test\ub294 fold\ubcc4 \ud3c9\uade0 \ucc28\uc774\ub97c \ubcf4\uace0, Wilcoxon\uc740 fold \uc218\uac00 5\uac1c\ub77c \ubcf4\uc218\uc801\uc73c\ub85c \ud574\uc11d\ud569\ub2c8\ub2e4. LightGBM/XGBoost\ub294 baseline \ub300\ube44 F1, recall, PR-AUC\ub97c \uac1c\uc120\ud558\uace0 false negative rate\ub97c \ub0ae\ucd94\uc5c8\uc2b5\ub2c8\ub2e4."))

        metric = st.selectbox(ko(r"\ube44\uad50 \uc9c0\ud45c"), ["f1", "recall", "pr_auc", "roc_auc", "false_negative_rate"])
        st.plotly_chart(
            px.box(experiments, x="experiment", y=metric, points="all", title=f"5-fold {metric} distribution"),
            width="stretch",
        )

        st.subheader(ko(r"\uc2dc\ub3c4\ud55c \ubc29\ubc95\ub860\uacfc \ud6a8\uacfc \ud574\uc11d"))
        method_notes = pd.DataFrame(
            [
                [ko(r"\uc120\ud615 \ubca0\uc774\uc2a4\ub77c\uc778"), ko(r"\uc800\ud574\uc0c1\ub3c4 \uc774\ubbf8\uc9c0 \ud53c\ucc98\ub9cc\uc73c\ub85c \ubd84\ub958 \uac00\ub2a5\uc131 \ud655\uc778"), ko(r"\uc131\ub2a5\uc740 \uc88b\uc9c0\ub9cc tree \uacc4\uc5f4\ubcf4\ub2e4 false negative\uac00 \ub9ce\uc2b5\ub2c8\ub2e4.")],
                [ko(r"\ud074\ub798\uc2a4 \uac00\uc911\uce58"), ko(r"\ubd88\ub7c9/\uc815\uc0c1 \ube44\uc728 \ucc28\uc774 \ubcf4\uc815"), ko(r"\ud604\uc7ac \ub370\uc774\ud130\uc5d0\uc11c\ub294 class weight\ub9cc\uc73c\ub85c\ub294 \uac1c\uc120\uc774 \uc791\uc558\uc2b5\ub2c8\ub2e4.")],
                [ko(r"Random Forest"), ko(r"\ube44\uc120\ud615 \ud53d\uc140 \ud328\ud134 \ud3ec\ucc29"), ko(r"\uc120\ud615 \ubaa8\ub378\ubcf4\ub2e4 \uac1c\uc120\ub418\uc9c0\ub9cc boosting\ubcf4\ub2e4\ub294 \uc57d\ud588\uc2b5\ub2c8\ub2e4.")],
                [ko(r"LightGBM"), ko(r"\ud53d\uc140/\ubc1d\uae30/\ub300\ube44 \uc0c1\ud638\uc791\uc6a9 \ud559\uc2b5"), ko(r"\ucd5c\uace0 \uc131\ub2a5\uc785\ub2c8\ub2e4. \ubc1d\uae30/\ub300\ube44\uc640 \ud2b9\uc815 \uc601\uc5ed \ud53d\uc140 \ucc28\uc774\ub97c \ud568\uaed8 \ud65c\uc6a9\ud55c \uac83\uc73c\ub85c \ubcf4\uc785\ub2c8\ub2e4.")],
                [ko(r"XGBoost"), ko(r"LightGBM\uacfc \uc720\uc0ac\ud55c boosting \ube44\uad50\uad70"), ko(r"\uac15\ub825\ud558\uc9c0\ub9cc \uc774\ubc88 \uc2e4\ud5d8\uc5d0\uc11c\ub294 LightGBM\ubcf4\ub2e4 \uc624\ubd84\ub958 \uac10\uc18c \ud3ed\uc774 \uc791\uc558\uc2b5\ub2c8\ub2e4.")],
            ],
            columns=[ko(r"\ubc29\ubc95"), ko(r"\ubaa9\uc801"), ko(r"\ud574\uc11d")],
        )
        st.dataframe(method_notes, width="stretch")

        if not oof.empty:
            selected = st.selectbox("OOF curve model", sorted(oof["experiment"].unique()))
            model_oof = oof[oof["experiment"] == selected]
            fpr, tpr, _ = roc_curve(model_oof["label"], model_oof["prob_defect"])
            precision, recall, _ = precision_recall_curve(model_oof["label"], model_oof["prob_defect"])
            left, right = st.columns(2)
            left.plotly_chart(go.Figure(data=go.Scatter(x=fpr, y=tpr)).update_layout(title="OOF ROC Curve", xaxis_title="FPR", yaxis_title="TPR"), width="stretch")
            right.plotly_chart(go.Figure(data=go.Scatter(x=recall, y=precision)).update_layout(title="OOF PR Curve", xaxis_title="Recall", yaxis_title="Precision"), width="stretch")

        st.subheader(ko(r"\ud5a5\ud6c4 \ucd94\uac00 \uc2e4\ud5d8"))
        st.markdown(
            ko(
                r"""
- **CNN/transfer learning \ubcf8 \uc2e4\ud5d8**: `resnet18`, `mobilenet_v3_small`, `efficientnet_b0`\ub97c 5-fold CV\ub85c \ud559\uc2b5\ud558\uace0 GradCAM\uc73c\ub85c \uc2e4\uc81c \uacb0\ud568 \uc601\uc5ed\uc744 \ubcf4\ub294\uc9c0 \ud655\uc778\ud569\ub2c8\ub2e4.
- **Threshold \ucd5c\uc801\ud654**: \ud604\uc7a5 \ud488\uc9c8\uac80\uc0ac\uc5d0\uc11c\ub294 false negative \ube44\uc6a9\uc774 \ud06c\ubbc0\ub85c recall \uc6b0\uc120 threshold\ub97c \ube44\uad50\ud569\ub2c8\ub2e4.
- **Hard example mining**: \uace0\uc2e0\ub8b0 \uc624\ubd84\ub958\uc640 \uc784\uacc4\uce58 \uadfc\ucc98 \uc624\ubd84\ub958\ub97c \ub2e4\uc74c \ud559\uc2b5\uc5d0 \ub354 \ubc18\uc601\ud569\ub2c8\ub2e4.
- **\uc720\uc0ac \uc774\ubbf8\uc9c0 \ub204\uc218 \uc810\uac80**: perceptual hash\ub85c fold \uac04 \uc720\uc0ac \uc774\ubbf8\uc9c0 \uc11e\uc784\uc744 \uc904\uc785\ub2c8\ub2e4.
"""
            )
        )

with tab_features:
    st.header(ko(r"\uc785\ub825 \ubcc0\uc218 \ubd84\uc11d"))
    st.caption(ko(r"\ud604\uc7ac \ube60\ub978 \uc2e4\ud5d8\uc740 32x32 grayscale pixel feature\uc640 width/height/brightness/contrast\ub97c \uc0ac\uc6a9\ud588\uc2b5\ub2c8\ub2e4. CNN \uc2e4\ud5d8\uc5d0\uc11c\ub294 GradCAM\uc73c\ub85c \ub300\uccb4\ud558\uba74 \ub429\ub2c8\ub2e4."))
    if importance.empty or feature_stats.empty:
        st.info(ko(r"\uc785\ub825 \ubcc0\uc218 \ubd84\uc11d \ub9ac\ud3ec\ud2b8\uac00 \uc5c6\uc2b5\ub2c8\ub2e4. `python -m src.dashboard_insights`\ub97c \uc2e4\ud589\ud558\uc138\uc694."))
    else:
        insight = dashboard_insights.get("top_feature_summary", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Pixel importance share", f"{insight.get('pixel_importance_share', 0):.1%}")
        c2.metric("Metadata importance share", f"{insight.get('metadata_importance_share', 0):.1%}")
        c3.metric(ko(r"\uc0c1\uc704 \ud53d\uc140 \uc601\uc5ed \uc218"), f"{insight.get('top_pixel_count', 0)}")

        left, right = st.columns([1, 1])
        with left:
            st.subheader(ko(r"\uc0c1\uc704 \uc911\uc694 \ubcc0\uc218"))
            st.dataframe(importance.head(25), width="stretch")
        with right:
            st.subheader(ko(r"\ud53d\uc140 \uc911\uc694\ub3c4 heatmap"))
            st.plotly_chart(px.imshow(pixel_importance_grid(importance), color_continuous_scale="Viridis", title="LightGBM feature importance"), width="stretch")

        st.subheader(ko(r"\uc911\uc694 \ubcc0\uc218\uc758 \ud1b5\uacc4\uc801 \ucc28\uc774"))
        stats_view = feature_stats.copy()
        stats_view["ttest_pvalue"] = stats_view["ttest_pvalue"].map(format_pvalue)
        stats_view["mannwhitney_pvalue"] = stats_view["mannwhitney_pvalue"].map(format_pvalue)
        st.dataframe(stats_view, width="stretch")
        st.markdown(
            ko(
                r"""
- \ubd88\ub7c9 \uc774\ubbf8\uc9c0\ub294 \uc815\uc0c1 \uc774\ubbf8\uc9c0\ubcf4\ub2e4 \ud3c9\uade0\uc801\uc73c\ub85c **\ub354 \uc5b4\ub461\uace0 \ub300\ube44\uac00 \ub0ae\uc740 \uacbd\ud5a5**\uc774 \uc788\uc2b5\ub2c8\ub2e4.
- \uc911\uc694 \ud53d\uc140 \uc601\uc5ed\uacfc \ube44\uc911\uc694 \uc601\uc5ed \ubaa8\ub450\uc5d0\uc11c \uc815\uc0c1/\ubd88\ub7c9 intensity \ucc28\uc774\uac00 \uc720\uc758\ud558\uac8c \ub098\ud0c0\ub0ac\uc2b5\ub2c8\ub2e4.
- \uc989, \ubaa8\ub378\uc774 \uacb0\ud568 \uc790\uccb4\ubfd0 \uc544\ub2c8\ub77c \uc870\uba85/\ud45c\uba74 \uba85\uc554 \ud328\ud134\ub3c4 \ud568\uaed8 \uc0ac\uc6a9\ud560 \uac00\ub2a5\uc131\uc774 \uc788\uc2b5\ub2c8\ub2e4.
"""
            )
        )

        if not correlations.empty:
            st.subheader(ko(r"\uc0c1\uad00\uad00\uacc4 \ud0d0\uc0c9"))
            st.plotly_chart(px.imshow(correlations.set_index("variable"), text_auto=".2f", title="Feature / prediction correlation"), width="stretch")

with tab_errors:
    st.header(ko(r"\uc624\ubd84\ub958 \ub370\uc774\ud130 \uc2ec\ud654 \ubd84\uc11d"))
    if errors.empty:
        st.info(ko(r"\uc624\ubd84\ub958 \ubd84\uc11d \ub9ac\ud3ec\ud2b8\uac00 \uc5c6\uc2b5\ub2c8\ub2e4. `python -m src.dashboard_insights`\ub97c \uc2e4\ud589\ud558\uc138\uc694."))
    else:
        err_summary = dashboard_insights.get("error_summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(ko(r"\uc804\uccb4 \uc624\ubd84\ub958"), err_summary.get("total_errors", len(errors)))
        c2.metric("False negatives", err_summary.get("false_negatives", int((errors["error_type"] == "False Negative").sum())))
        c3.metric("False positives", err_summary.get("false_positives", int((errors["error_type"] == "False Positive").sum())))
        c4.metric(ko(r"\uace0\uc2e0\ub8b0 \uc624\ubd84\ub958"), err_summary.get("high_confidence_errors", int((errors["error_margin_group"] == "high_confidence").sum())))

        st.subheader(ko(r"\uc784\uacc4\uce58 \uadfc\ucc98 \uc624\ubd84\ub958 vs \uace0\ub9c8\uc9c4 \uc624\ubd84\ub958"))
        if not error_groups.empty:
            st.dataframe(error_groups, width="stretch")
            st.plotly_chart(px.bar(error_groups, x="error_margin_group", y=["false_negative", "false_positive"], barmode="group", title="Error type by margin group"), width="stretch")

        st.markdown(
            ko(
                r"""
- **near_threshold** \ucf00\uc774\uc2a4\ub294 \ubaa8\ub378\ub3c4 \uc560\ub9e4\ud558\uac8c \ubcf8 \uc0ac\ub840\ub77c threshold \uc870\uc815, augmentation, \ucd94\uac00 \ub77c\ubca8 \uac80\ud1a0\ub85c \uac1c\uc120\ub420 \uac00\ub2a5\uc131\uc774 \ud07d\ub2c8\ub2e4.
- **high_confidence** \ucf00\uc774\uc2a4\ub294 \ub77c\ubca8 \uc624\ub958, \uc774\ubbf8\uc9c0 \uc774\uc0c1\uce58, \uc870\uba85/\ubc30\uacbd shortcut\uc744 \uc6b0\uc120 \uc758\uc2ec\ud574\uc57c \ud569\ub2c8\ub2e4.
"""
            )
        )

        st.subheader(ko(r"\uc624\ubd84\ub958 \uc774\ubbf8\uc9c0 \uc0d8\ud50c \uac80\ud1a0"))
        group_choice = st.radio(ko(r"\uc624\ubd84\ub958 \uadf8\ub8f9"), ["all", "near_threshold", "high_confidence"], horizontal=True)
        type_choice = st.radio(ko(r"\uc624\ub958 \uc720\ud615"), ["all", "False Negative", "False Positive"], horizontal=True)
        view = errors.copy()
        if group_choice != "all":
            view = view[view["error_margin_group"] == group_choice]
        if type_choice != "all":
            view = view[view["error_type"] == type_choice]
        view = view.sort_values("margin_from_threshold", ascending=False)
        st.dataframe(
            view[["relative_path", "label", "prediction", "prob_defect", "margin_from_threshold", "error_type", "error_margin_group", "label_issue_risk", "brightness", "contrast"]].head(50),
            width="stretch",
        )

        max_images = st.slider(ko(r"\ud45c\uc2dc\ud560 \uc774\ubbf8\uc9c0 \uc218"), 4, 24, 12, step=4)
        for start in range(0, min(len(view), max_images), 4):
            cols = st.columns(4)
            for col, (_, row) in zip(cols, view.iloc[start : start + 4].iterrows()):
                label = "Defective" if row["label"] == 1 else "OK"
                pred = "Defective" if row["prediction"] == 1 else "OK"
                caption = f"{row['error_type']} | true={label} / pred={pred} | p={row['prob_defect']:.3f} | margin={row['margin_from_threshold']:.3f}"
                image_path = resolve_image_path(row)
                if image_path is None:
                    col.info(ko(r"\uc774\ubbf8\uc9c0 \ud30c\uc77c\uc740 Cloud \ubc30\ud3ec\uc5d0 \ud3ec\ud568\ub418\uc9c0 \uc54a\uc558\uc2b5\ub2c8\ub2e4."))
                    col.caption(caption)
                else:
                    col.image(str(image_path), caption=caption, width="stretch")

        st.subheader(ko(r"\ub77c\ubca8 \uc624\ub958 \uac00\ub2a5\uc131 \uc870\uc0ac"))
        risky = errors[errors["label_issue_risk"] == "review_label_or_outlier"].sort_values("margin_from_threshold", ascending=False)
        st.write(ko(r"\ub77c\ubca8 \ub610\ub294 \uc774\uc0c1\uce58 \uc7ac\uac80\ud1a0 \uc6b0\uc120 \ud6c4\ubcf4: ") + f"{len(risky)}")
        st.dataframe(risky[["relative_path", "error_type", "prob_defect", "margin_from_threshold", "brightness", "contrast"]].head(30), width="stretch")

        st.subheader(ko(r"\uc778\uc0ac\uc774\ud2b8 \uae30\ubc18 \ubaa8\ub378 \uace0\ub3c4\ud654 \uacc4\ud68d"))
        st.markdown(
            ko(
                r"""
1. **\ub77c\ubca8 \uac80\uc218 \ub8e8\ud504**: \uace0\ub9c8\uc9c4 \uc624\ubd84\ub958\ub97c \uc6b0\uc120 \uac80\ud1a0\ud574 \ub77c\ubca8 \uc624\ub958, \uc774\ubbf8\uc9c0 \ud488\uc9c8 \ubb38\uc81c, \uc720\uc0ac \uc774\ubbf8\uc9c0 \ub204\uc218\ub97c \uccb4\ud06c\ud569\ub2c8\ub2e4.
2. **\uc6b4\uc601 threshold \uc7ac\uc124\uc815**: false negative\ub97c \uc904\uc774\ub294 recall-first threshold\ub97c \ubcc4\ub3c4 \uc6b4\uc601\uc548\uc73c\ub85c \ube44\uad50\ud569\ub2c8\ub2e4.
3. **CNN + GradCAM \uac80\uc99d**: transfer learning \ubaa8\ub378\uc744 \ud559\uc2b5\ud558\uace0 \uc2e4\uc81c \uacb0\ud568 \uc601\uc5ed\uc744 \ubcf4\ub294\uc9c0 \ud655\uc778\ud569\ub2c8\ub2e4.
4. **\uc870\uba85 \uac15\uac74\ud654 \uc2e4\ud5d8**: ColorJitter, histogram equalization, normalization \uc870\ud569\uc744 ablation\ud569\ub2c8\ub2e4.
5. **Hard example mining**: near-threshold\uc640 high-confidence error\ub97c sampler \ub610\ub294 \uac00\uc911\uce58\ub85c \ubc18\uc601\ud569\ub2c8\ub2e4.
"""
            )
        )
