from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42
FIXED_CAMPAIGN_COST_A = 100.0


def log_step(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}", flush=True)


def save_new(df: pd.DataFrame, filename: str) -> None:
    path = OUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Wrote {path}", flush=True)
    print(df.to_string(index=False), flush=True)


def top_decile_lift(y_true: np.ndarray, y_score: np.ndarray) -> float:
    data = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False)
    n = max(1, math.ceil(len(data) * 0.10))
    base = data["y"].mean()
    return float(data.head(n)["y"].mean() / base) if base > 0 else np.nan


def make_preprocessor(df: pd.DataFrame, target: str, drop_cols: set[str]) -> tuple[ColumnTransformer, list[str]]:
    features = [c for c in df.columns if c not in drop_cols | {target}]
    num_cols = [c for c in features if is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return pre, features


def split_xy(df: pd.DataFrame, target: str, drop_cols: set[str], time_col: str = "LastPurchaseDate"):
    pre, features = make_preprocessor(df, target, drop_cols)
    X = df[features].copy()
    y = df[target].astype(int)
    if time_col in df.columns:
        dates = pd.to_datetime(df[time_col])
        cutoff = dates.quantile(0.8)
        train_idx = dates <= cutoff
        if train_idx.nunique() == 2 and y[train_idx].nunique() == 2 and y[~train_idx].nunique() == 2:
            return X[train_idx], X[~train_idx], y[train_idx], y[~train_idx], pre
    return (*train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE), pre)


def model_defs(pos_weight: float) -> dict[str, object]:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    }
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.06,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    return models


def evaluate_models(df: pd.DataFrame, target: str, dataset: str, output_csv: str, drop_cols: set[str]):
    X_train, X_test, y_train, y_test, pre = split_xy(df, target, drop_cols)
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    feature_names = pre.get_feature_names_out()
    pos_weight = max(1.0, (len(y_train) - y_train.sum()) / max(1, y_train.sum()))
    rows = []
    fitted = {}
    for imbalance in ["class_weight", "SMOTE"]:
        if imbalance == "SMOTE":
            minority = int(y_train.value_counts().min())
            if minority < 2:
                rows.append({"dataset": dataset, "model": "ALL", "imbalance": imbalance, "error": "not enough minority samples"})
                continue
            sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, minority - 1))
            X_fit, y_fit = sampler.fit_resample(X_train_t, y_train)
        else:
            X_fit, y_fit = X_train_t, y_train
        for name, model in model_defs(pos_weight).items():
            try:
                if imbalance == "SMOTE" and hasattr(model, "class_weight"):
                    try:
                        model.set_params(class_weight=None)
                    except Exception:
                        pass
                model.fit(X_fit, y_fit)
                score = model.predict_proba(X_test_t)[:, 1]
                rows.append(
                    {
                        "dataset": dataset,
                        "model": name,
                        "imbalance": imbalance,
                        "auc_roc": roc_auc_score(y_test, score),
                        "pr_auc": average_precision_score(y_test, score),
                        "top_decile_lift": top_decile_lift(y_test.to_numpy(), score),
                        "train_rows": len(y_train),
                        "test_rows": len(y_test),
                        "target_rate_train": y_train.mean(),
                        "target_rate_test": y_test.mean(),
                    }
                )
                fitted[(name, imbalance)] = (model, pre, X_train_t, X_test_t, X_test, y_test, score, feature_names)
            except Exception as exc:
                rows.append({"dataset": dataset, "model": name, "imbalance": imbalance, "error": str(exc)})
    perf = pd.DataFrame(rows).sort_values(["auc_roc", "pr_auc"], ascending=False, na_position="last")
    save_new(perf, output_csv)
    best_row = perf.dropna(subset=["auc_roc"]).iloc[0]
    best = fitted[(best_row["model"], best_row["imbalance"])]
    return perf, best


def normalize_shap(raw):
    arr = raw
    if isinstance(arr, list):
        arr = arr[1] if len(arr) > 1 else arr[0]
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[2] == 2:
            arr = arr[:, :, 1]
        elif arr.shape[0] == 2:
            arr = arr[1, :, :]
        else:
            arr = arr[:, :, 0]
    return arr


def shap_outputs(best, top_csv: str, summary_png: str) -> pd.DataFrame:
    model, _, X_train_t, X_test_t, _, _, _, feature_names = best
    background = X_train_t[: min(500, X_train_t.shape[0])]
    sample = X_test_t[: min(500, X_test_t.shape[0])]
    try:
        explainer = shap.Explainer(model, background, feature_names=feature_names)
        values = explainer(sample)
        shap_values = normalize_shap(values.values)
    except Exception:
        explainer = shap.TreeExplainer(model)
        shap_values = normalize_shap(explainer.shap_values(sample))

    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / summary_png, dpi=160, bbox_inches="tight")
    plt.close()

    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).head(10)
    save_new(imp, top_csv)
    return imp


def recalibrate_bgnbd(labeled: pd.DataFrame) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    palive = labeled["p_alive"].astype(float)
    churn_rate = labeled["churn_90d"].mean()
    labeled = labeled.copy()
    churn_n = int(round(len(labeled) * churn_rate))
    ranked_idx = labeled.sort_values(["p_alive", "CustomerID"], ascending=[True, True]).head(churn_n).index
    labeled["bgnbd_churn_recalibrated"] = 0
    labeled.loc[ranked_idx, "bgnbd_churn_recalibrated"] = 1
    threshold = float(labeled.loc[ranked_idx, "p_alive"].max())

    desc = palive.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.30, churn_rate, 0.50, 0.75, 0.90, 0.95, 0.99])
    desc_df = desc.reset_index()
    desc_df.columns = ["statistic", "p_alive"]
    save_new(desc_df, "online_bgnbd_palive_descriptive_stats.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(palive, bins=50, ax=ax)
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.4f}")
    ax.set_title("BG/NBD P(alive) distribution")
    ax.set_xlabel("P(alive)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "online_bgnbd_palive_histogram.png", dpi=160)
    plt.close(fig)

    cm = confusion_matrix(labeled["churn_90d"], labeled["bgnbd_churn_recalibrated"])
    summary = pd.DataFrame(
        [
            ["churn_90d_rate", churn_rate],
            ["p_alive_threshold_percentile", churn_rate],
            ["p_alive_recalibrated_threshold", threshold],
            ["rank_selected_churn_customers", churn_n],
            ["bgnbd_recalibrated_churn_rate", labeled["bgnbd_churn_recalibrated"].mean()],
            ["cohens_kappa_90d_vs_recalibrated_bgnbd", cohen_kappa_score(labeled["churn_90d"], labeled["bgnbd_churn_recalibrated"])],
            ["confusion_matrix_90d_vs_recalibrated_bgnbd", json.dumps(cm.tolist())],
            ["previous_fixed_threshold_0_5_churn_rate", (labeled["p_alive"] < 0.5).mean()],
        ],
        columns=["metric", "value"],
    )
    save_new(summary, "online_labeling_summary_v2.csv")
    return labeled, threshold, summary


def write_labeling_note(threshold: float, summary: pd.DataFrame) -> None:
    values = dict(zip(summary["metric"], summary["value"]))
    note = f"""# BG/NBD Label Recalibration Note

The fixed `P(alive) < 0.5` rule was not usable for this Online Retail run because it classified 0.00% of customers as churn. The 90-day inactivity label has a churn rate of {float(values['churn_90d_rate']):.2%}, so the recalibrated rule uses the lower {float(values['p_alive_threshold_percentile']):.2%} of the BG/NBD `P(alive)` distribution as churn. Because `P(alive)` is extremely concentrated near 1.0, the final assignment is rank-based rather than a plain `<= quantile` comparison; this avoids classifying all tied-at-threshold customers as churn. The selected cutoff value is `P(alive) <= {threshold:.6f}` for the rank-selected group.

With this distribution-based cutoff, the BG/NBD-derived churn rate is {float(values['bgnbd_recalibrated_churn_rate']):.2%}. Agreement with the 90-day inactivity label remains limited: Cohen's kappa is {float(values['cohens_kappa_90d_vs_recalibrated_bgnbd']):.3f}, with confusion matrix {values['confusion_matrix_90d_vs_recalibrated_bgnbd']}. This means the two labels identify different customer-risk concepts even after matching churn prevalence: the 90-day label is a direct inactivity rule, while BG/NBD ranks customers by model-based alive probability.
"""
    (OUT_DIR / "online_labeling_recalibration_note.md").write_text(note, encoding="utf-8")


def neslin_curve(scores: np.ndarray, values: np.ndarray, delta: float, kappa: float, gamma: float) -> pd.DataFrame:
    order = np.argsort(-scores)
    rows = []
    for alpha in np.linspace(0, 1, 101):
        n = int(round(len(order) * alpha))
        if n == 0:
            rows.append([alpha, n, -FIXED_CAMPAIGN_COST_A])
            continue
        idx = order[:n]
        beta = scores[idx]
        v = values[idx]
        per_customer = beta * gamma * (v - delta - kappa) + beta * (1 - gamma) * (-kappa) + (1 - beta) * (-delta - kappa)
        rows.append([alpha, n, float(per_customer.sum() - FIXED_CAMPAIGN_COST_A)])
    return pd.DataFrame(rows, columns=["target_ratio_alpha", "targeted_customers", "neslin_profit"])


def neslin_sensitivity(best) -> pd.DataFrame:
    model, pre, _, _, X_test, _, score, _ = best
    scores = model.predict_proba(pre.transform(X_test))[:, 1]
    values = X_test["Monetary"].to_numpy() if "Monetary" in X_test.columns else np.repeat(1.0, len(X_test))
    combos = [
        (5, 0.5, 0.25),
        (15, 0.5, 0.25),
        (30, 1, 0.25),
        (50, 1, 0.25),
        (30, 2, 0.25),
        (50, 2, 0.25),
        (15, 1, 0.15),
        (30, 1, 0.15),
        (50, 1, 0.15),
        (15, 1, 0.40),
        (30, 1, 0.40),
        (50, 2, 0.40),
    ]
    rows = []
    curves = {}
    for delta, kappa, gamma in combos:
        curve = neslin_curve(scores, values, delta, kappa, gamma)
        opt = curve.loc[curve["neslin_profit"].idxmax()]
        key = (delta, kappa, gamma)
        curves[key] = curve
        rows.append(
            {
                "delta": delta,
                "kappa": kappa,
                "gamma": gamma,
                "optimal_alpha": opt["target_ratio_alpha"],
                "max_profit": opt["neslin_profit"],
                "is_interior_optimum": 0 < opt["target_ratio_alpha"] < 1,
            }
        )
    summary = pd.DataFrame(rows)
    save_new(summary, "online_neslin_sensitivity.csv")

    interior = summary[summary["is_interior_optimum"]].sort_values("max_profit", ascending=False)
    if len(interior):
        chosen = interior.iloc[0]
    else:
        summary["distance_from_boundary"] = np.minimum(summary["optimal_alpha"], 1 - summary["optimal_alpha"])
        chosen = summary.sort_values(["distance_from_boundary", "max_profit"], ascending=False).iloc[0]
    key = (chosen["delta"], chosen["kappa"], chosen["gamma"])
    curve = curves[key]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["target_ratio_alpha"], curve["neslin_profit"])
    ax.axvline(chosen["optimal_alpha"], color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Neslin sensitivity curve: delta={key[0]}, kappa={key[1]}, gamma={key[2]}")
    ax.set_xlabel("Target ratio alpha")
    ax.set_ylabel("Profit")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "online_neslin_profit_curve_v2.png", dpi=160)
    plt.close(fig)
    return summary


def write_neslin_note(summary: pd.DataFrame) -> None:
    interiors = summary[summary["is_interior_optimum"]]
    baseline = summary[(summary["delta"] == 5) & (summary["kappa"] == 0.5) & (summary["gamma"] == 0.25)].iloc[0]
    high_cost = summary.sort_values(["delta", "kappa", "gamma"], ascending=[False, False, False]).iloc[0]
    if len(interiors):
        interior_text = (
            f"Interior optima were observed in {len(interiors)} of {len(summary)} tested scenarios. "
            f"The highest-profit interior scenario was delta={interiors.sort_values('max_profit', ascending=False).iloc[0]['delta']}, "
            f"kappa={interiors.sort_values('max_profit', ascending=False).iloc[0]['kappa']}, "
            f"gamma={interiors.sort_values('max_profit', ascending=False).iloc[0]['gamma']}."
        )
    else:
        interior_text = (
            "No strict interior optimum was observed on the tested alpha grid; the plotted v2 curve uses the scenario closest to an interior solution. "
            "This indicates that the current score ranking and Monetary-based value proxy still make marginal targeted customers profitable under the tested assumptions, or unprofitable enough that the optimum sits at a boundary."
        )
    note = f"""# Neslin Sensitivity Note

The baseline scenario delta={baseline['delta']}, kappa={baseline['kappa']}, gamma={baseline['gamma']} has optimal alpha={baseline['optimal_alpha']:.2f} and max profit={baseline['max_profit']:.2f}. In the higher-cost scenario delta={high_cost['delta']}, kappa={high_cost['kappa']}, gamma={high_cost['gamma']}, optimal alpha={high_cost['optimal_alpha']:.2f} and max profit={high_cost['max_profit']:.2f}.

{interior_text} In general, increasing delta and kappa reduces the profitability of lower-ranked customers first, so alpha should decrease when campaign costs rise unless higher gamma or very high Monetary values compensate for the additional cost.
"""
    (OUT_DIR / "online_neslin_sensitivity_note.md").write_text(note, encoding="utf-8")


def leakage_comparison(no_recency: pd.DataFrame, bgnbd: pd.DataFrame | None) -> pd.DataFrame:
    old = pd.read_csv(OUT_DIR / "online_model_performance.csv")
    rows = []
    for label, df in [("original_leaky_recency", old), ("no_recency_90d_label", no_recency)]:
        best = df.dropna(subset=["auc_roc"]).sort_values(["auc_roc", "pr_auc"], ascending=False).iloc[0]
        rows.append(
            {
                "experiment": label,
                "best_model": best["model"],
                "imbalance": best["imbalance"],
                "auc_roc": best["auc_roc"],
                "pr_auc": best["pr_auc"],
                "top_decile_lift": best["top_decile_lift"],
            }
        )
    if bgnbd is not None and "auc_roc" in bgnbd.columns and bgnbd["auc_roc"].notna().any():
        best = bgnbd.dropna(subset=["auc_roc"]).sort_values(["auc_roc", "pr_auc"], ascending=False).iloc[0]
        rows.append(
            {
                "experiment": "bgnbd_recalibrated_label",
                "best_model": best["model"],
                "imbalance": best["imbalance"],
                "auc_roc": best["auc_roc"],
                "pr_auc": best["pr_auc"],
                "top_decile_lift": best["top_decile_lift"],
            }
        )
    comp = pd.DataFrame(rows)
    comp["auc_drop_vs_original"] = comp["auc_roc"].iloc[0] - comp["auc_roc"]
    save_new(comp, "online_leakage_comparison.csv")
    return comp


def append_results_section(comp: pd.DataFrame, label_summary: pd.DataFrame, neslin_summary: pd.DataFrame) -> None:
    no_rec = comp[comp["experiment"] == "no_recency_90d_label"].iloc[0]
    bgnbd = comp[comp["experiment"] == "bgnbd_recalibrated_label"].iloc[0] if (comp["experiment"] == "bgnbd_recalibrated_label").any() else None
    label_values = dict(zip(label_summary["metric"], label_summary["value"]))
    interiors = int(neslin_summary["is_interior_optimum"].sum())
    section = f"""

## 11. Leakage Correction & Sensitivity Analysis

The leakage-corrected A-1 experiment removed `Recency` while keeping Frequency, Monetary, AvgOrderValue, TenureDays, and Country. The best no-recency model was {no_rec['best_model']} + {no_rec['imbalance']} with AUC-ROC {no_rec['auc_roc']:.3f}, PR-AUC {no_rec['pr_auc']:.3f}, and top-decile lift {no_rec['top_decile_lift']:.3f}. Compared with the original leaky AUC of 1.000, the AUC drop is {no_rec['auc_drop_vs_original']:.3f}, confirming that the previous perfect score was label leakage rather than pure predictive signal.

BG/NBD recalibration used the lower {float(label_values['p_alive_threshold_percentile']):.2%} percentile of `P(alive)` as the churn cutoff, yielding threshold {float(label_values['p_alive_recalibrated_threshold']):.6f}. The recalibrated BG/NBD churn rate is {float(label_values['bgnbd_recalibrated_churn_rate']):.2%}; agreement with the 90-day label remains weak with kappa {float(label_values['cohens_kappa_90d_vs_recalibrated_bgnbd']):.3f}. {f"The A-2 BG/NBD-label model achieved AUC-ROC {bgnbd['auc_roc']:.3f} and PR-AUC {bgnbd['pr_auc']:.3f}." if bgnbd is not None else "The A-2 BG/NBD-label model was not available."}

Neslin sensitivity tested {len(neslin_summary)} cost/retention scenarios. Strict interior optima appeared in {interiors} scenarios. Detailed outputs are in `online_model_performance_no_recency.csv`, `online_model_performance_bgnbd_label.csv`, `online_labeling_summary_v2.csv`, `online_neslin_sensitivity.csv`, and the accompanying note files.
"""
    path = OUT_DIR / "RESULTS.md"
    text = path.read_text(encoding="utf-8")
    marker = "\n## 11. Leakage Correction & Sensitivity Analysis"
    if marker in text:
        text = text[: text.index(marker)]
    path.write_text(text.rstrip() + section, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    labeled_path = OUT_DIR / "online_customer_rfm_labels.csv"
    if not labeled_path.exists():
        raise FileNotFoundError("Run scripts/run_churn_pipeline.py first; online_customer_rfm_labels.csv is required.")
    labeled = pd.read_csv(labeled_path)
    labeled["LastPurchaseDate"] = pd.to_datetime(labeled["LastPurchaseDate"])

    base_drop = {"CustomerID", "LastPurchaseDate", "p_alive", "bgnbd_churn", "bgnbd_churn_recalibrated", "churn_90d"}

    log_step("Experiment B: BG/NBD threshold recalibration")
    labeled, threshold, label_summary = recalibrate_bgnbd(labeled)
    write_labeling_note(threshold, label_summary)

    log_step("Experiment A-1: no Recency model")
    no_recency_df = labeled.drop(columns=["Recency"]).copy()
    no_recency_perf, no_recency_best = evaluate_models(
        no_recency_df,
        "churn_90d",
        "Online_NoRecency",
        "online_model_performance_no_recency.csv",
        drop_cols=base_drop - {"churn_90d"},
    )
    shap_outputs(no_recency_best, "online_shap_top10_no_recency.csv", "online_shap_summary_no_recency.png")

    log_step("Experiment A-2: recalibrated BG/NBD label model")
    bgnbd_drop = {"CustomerID", "LastPurchaseDate", "p_alive", "bgnbd_churn", "churn_90d", "bgnbd_churn_recalibrated"}
    bgnbd_perf, _ = evaluate_models(
        labeled,
        "bgnbd_churn_recalibrated",
        "Online_BGNBdLabel",
        "online_model_performance_bgnbd_label.csv",
        drop_cols=bgnbd_drop,
    )

    log_step("Leakage comparison")
    comp = leakage_comparison(no_recency_perf, bgnbd_perf)

    log_step("Experiment C: Neslin sensitivity")
    neslin_summary = neslin_sensitivity(no_recency_best)
    write_neslin_note(neslin_summary)

    log_step("Append RESULTS.md")
    append_results_section(comp, label_summary, neslin_summary)
    print("Done. Additional experiment outputs are in outputs/.", flush=True)


if __name__ == "__main__":
    main()
