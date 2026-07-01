from __future__ import annotations

import json
import math
import shutil
import warnings
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import is_numeric_dtype
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

from harness_validation import demo_cases


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"

# Neslin/EMP assumptions. These are report assumptions, not observed campaign costs.
INCENTIVE_COST_DELTA = 5.0
CONTACT_COST_KAPPA = 0.5
RETENTION_RATE_GAMMA = 0.25
FIXED_CAMPAIGN_COST_A = 100.0
CHURN_THRESHOLD_DAYS = 90
RANDOM_STATE = 42

UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
TELCO_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]


def log_step(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}", flush=True)


def save_df(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False), flush=True)


def download_file(url: str, target: Path) -> bool:
    if target.exists() and target.stat().st_size > 0:
        return True
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with target.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as exc:
        print(f"DOWNLOAD_FAILED url={url} error={exc}", flush=True)
        if target.exists():
            target.unlink()
        return False


def ensure_data() -> tuple[Path, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    online_xlsx = RAW_DIR / "Online Retail.xlsx"
    if not online_xlsx.exists():
        zip_path = RAW_DIR / "online_retail.zip"
        if not download_file(UCI_URL, zip_path):
            raise RuntimeError("Could not download UCI Online Retail dataset.")
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if member.lower().endswith(".xlsx"):
                    with zf.open(member) as src, online_xlsx.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    break
    print(f"Online Retail file: {online_xlsx}", flush=True)

    telco_csv = RAW_DIR / "Telco-Customer-Churn.csv"
    if not telco_csv.exists():
        ok = False
        for url in TELCO_URLS:
            ok = download_file(url, telco_csv)
            if ok:
                break
        if not ok:
            raise RuntimeError("Could not download Telco Customer Churn dataset.")
    print(f"Telco file: {telco_csv}", flush=True)
    return online_xlsx, telco_csv


def profile_online(df: pd.DataFrame) -> pd.DataFrame:
    customer_missing = df["CustomerID"].isna().mean()
    cancel_ratio = df["InvoiceNo"].astype(str).str.startswith("C").mean()
    profile = pd.DataFrame(
        [
            ["rows", len(df)],
            ["columns", df.shape[1]],
            ["date_min", df["InvoiceDate"].min()],
            ["date_max", df["InvoiceDate"].max()],
            ["customer_id_missing_ratio", customer_missing],
            ["cancel_invoice_ratio", cancel_ratio],
            ["quantity_nonpositive_ratio", (df["Quantity"] <= 0).mean()],
            ["unitprice_nonpositive_ratio", (df["UnitPrice"] <= 0).mean()],
        ],
        columns=["metric", "value"],
    )
    return profile


def profile_telco(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append([col, str(df[col].dtype), int(df[col].isna().sum()), float(df[col].isna().mean())])
    return pd.DataFrame(rows, columns=["column", "dtype", "missing_count", "missing_ratio"])


def preprocess_online(df: pd.DataFrame) -> pd.DataFrame:
    original = len(df)
    cancel = df["InvoiceNo"].astype(str).str.startswith("C")
    missing_customer = df["CustomerID"].isna()
    bad_qty = df["Quantity"] <= 0
    bad_price = df["UnitPrice"] <= 0
    removal = pd.DataFrame(
        [
            ["starts_with_C_cancel_invoice", int(cancel.sum()), cancel.mean()],
            ["missing_customer_id", int(missing_customer.sum()), missing_customer.mean()],
            ["quantity_le_0", int(bad_qty.sum()), bad_qty.mean()],
            ["unitprice_le_0", int(bad_price.sum()), bad_price.mean()],
        ],
        columns=["reason", "rows_matching_reason", "ratio_of_raw_rows"],
    )
    keep = ~(cancel | missing_customer | bad_qty | bad_price)
    clean = df.loc[keep].copy()
    summary = pd.concat(
        [
            pd.DataFrame([["raw_rows", original, 1.0], ["clean_rows", len(clean), len(clean) / original]], columns=removal.columns),
            removal,
        ],
        ignore_index=True,
    )
    save_df(summary, "online_preprocessing_summary.csv")
    clean["CustomerID"] = clean["CustomerID"].astype(int).astype(str)
    clean["Sales"] = clean["Quantity"] * clean["UnitPrice"]
    return clean


def build_rfm(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    snapshot = clean["InvoiceDate"].max() + pd.Timedelta(days=1)
    orders = clean.groupby(["CustomerID", "InvoiceNo"], as_index=False).agg(
        InvoiceDate=("InvoiceDate", "max"),
        Sales=("Sales", "sum"),
        Country=("Country", "first"),
    )
    rfm = orders.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Sales", "sum"),
        AvgOrderValue=("Sales", "mean"),
        TenureDays=("InvoiceDate", lambda s: (s.max() - s.min()).days + 1),
        LastPurchaseDate=("InvoiceDate", "max"),
        Country=("Country", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
    ).reset_index()
    desc = rfm[["Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays"]].describe().T.reset_index(names="feature")
    save_df(desc, "online_rfm_descriptive_stats.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
        sns.histplot(np.log1p(rfm[col]), bins=40, ax=ax)
        ax.set_title(f"log1p({col})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "online_rfm_histograms.png", dpi=160)
    plt.close(fig)
    return rfm, snapshot


def label_online(clean: pd.DataFrame, rfm: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    labeled = rfm.copy()
    labeled["churn_90d"] = (labeled["Recency"] > CHURN_THRESHOLD_DAYS).astype(int)
    try:
        tx = clean[["CustomerID", "InvoiceDate", "Sales"]].copy()
        summary = summary_data_from_transaction_data(
            tx, "CustomerID", "InvoiceDate", monetary_value_col="Sales", observation_period_end=snapshot
        )
        bgf = None
        last_error = None
        used_penalizer = None
        for penalizer in [0.01, 0.1, 1.0, 5.0]:
            try:
                candidate = BetaGeoFitter(penalizer_coef=penalizer)
                candidate.fit(summary["frequency"], summary["recency"], summary["T"])
                bgf = candidate
                used_penalizer = penalizer
                break
            except Exception as exc:
                last_error = exc
        if bgf is None:
            raise RuntimeError(last_error)
        summary["p_alive"] = bgf.conditional_probability_alive(summary["frequency"], summary["recency"], summary["T"])
        palive = summary["p_alive"].reset_index().rename(columns={"index": "CustomerID"})
        palive["CustomerID"] = palive["CustomerID"].astype(str)
        labeled = labeled.merge(palive, on="CustomerID", how="left")
        labeled["bgnbd_churn"] = (labeled["p_alive"] < 0.5).astype(int)
        bg_note = f"BG/NBD fit succeeded with penalizer_coef={used_penalizer}."
    except Exception as exc:
        labeled["p_alive"] = np.nan
        labeled["bgnbd_churn"] = labeled["churn_90d"]
        bg_note = f"BG/NBD fit failed; fallback used churn_90d. Error: {exc}"

    cm = confusion_matrix(labeled["churn_90d"], labeled["bgnbd_churn"])
    label_summary = pd.DataFrame(
        [
            ["snapshot_date", snapshot.date().isoformat()],
            ["churn_90d_rate", labeled["churn_90d"].mean()],
            ["bgnbd_churn_rate", labeled["bgnbd_churn"].mean()],
            ["cohens_kappa", cohen_kappa_score(labeled["churn_90d"], labeled["bgnbd_churn"])],
            ["confusion_matrix_90d_vs_bgnbd", json.dumps(cm.tolist())],
            ["bgnbd_note", bg_note],
        ],
        columns=["metric", "value"],
    )
    save_df(label_summary, "online_labeling_summary.csv")
    return labeled


def top_decile_lift(y_true: np.ndarray, y_score: np.ndarray) -> float:
    data = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False)
    n = max(1, math.ceil(len(data) * 0.10))
    base = data["y"].mean()
    return float(data.head(n)["y"].mean() / base) if base > 0 else np.nan


def make_preprocessor(df: pd.DataFrame, target: str) -> tuple[ColumnTransformer, list[str], list[str]]:
    drop_cols = {"CustomerID", target, "LastPurchaseDate", "p_alive", "bgnbd_churn"}
    features = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in features if is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return pre, num_cols, cat_cols


def split_xy(df: pd.DataFrame, target: str, time_col: str | None = None):
    pre, num_cols, cat_cols = make_preprocessor(df, target)
    drop_cols = {"CustomerID", target, "LastPurchaseDate", "p_alive", "bgnbd_churn"}
    X = df[[c for c in df.columns if c not in drop_cols]].copy()
    y = df[target].astype(int)
    if time_col and time_col in df.columns:
        cutoff = df[time_col].quantile(0.8)
        train_idx = df[time_col] <= cutoff
        if train_idx.nunique() == 2 and y[train_idx].nunique() == 2 and y[~train_idx].nunique() == 2:
            return X[train_idx], X[~train_idx], y[train_idx], y[~train_idx], pre, num_cols, cat_cols
    return (*train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE), pre, num_cols, cat_cols)


def model_defs(pos_weight: float) -> dict[str, object]:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=220, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
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


def evaluate_models(df: pd.DataFrame, target: str, dataset_name: str, time_col: str | None = None):
    X_train, X_test, y_train, y_test, pre, _, _ = split_xy(df, target, time_col)
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    feature_names = pre.get_feature_names_out()
    pos_weight = max(1.0, (len(y_train) - y_train.sum()) / max(1, y_train.sum()))
    results = []
    fitted = {}
    for imbalance in ["class_weight", "SMOTE"]:
        if imbalance == "SMOTE":
            k = min(5, int(y_train.value_counts().min()) - 1)
            if k < 1:
                print(f"SMOTE skipped for {dataset_name}: not enough minority samples", flush=True)
                continue
            sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
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
                row = {
                    "dataset": dataset_name,
                    "model": name,
                    "imbalance": imbalance,
                    "auc_roc": roc_auc_score(y_test, score),
                    "pr_auc": average_precision_score(y_test, score),
                    "top_decile_lift": top_decile_lift(y_test.to_numpy(), score),
                    "train_rows": len(y_train),
                    "test_rows": len(y_test),
                }
                results.append(row)
                fitted[(name, imbalance)] = (model, pre, X_train_t, X_test_t, y_test, score, feature_names)
            except Exception as exc:
                results.append({"dataset": dataset_name, "model": name, "imbalance": imbalance, "error": str(exc)})
    perf = pd.DataFrame(results).sort_values(["auc_roc", "pr_auc"], ascending=False, na_position="last")
    save_df(perf, f"{dataset_name.lower()}_model_performance.csv")
    best_row = perf.dropna(subset=["auc_roc"]).iloc[0]
    best = fitted[(best_row["model"], best_row["imbalance"])]
    return perf, best, X_test, y_test


def shap_plots(best, dataset_name: str) -> pd.DataFrame:
    model, pre, X_train_t, X_test_t, _, _, feature_names = best
    sample = X_test_t[: min(500, X_test_t.shape[0])]
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

    try:
        explainer = shap.Explainer(model, X_train_t[: min(500, X_train_t.shape[0])], feature_names=feature_names)
        values = explainer(sample)
        shap_values = normalize_shap(values.values)
    except Exception:
        explainer = shap.TreeExplainer(model)
        shap_values = normalize_shap(explainer.shap_values(sample))
    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{dataset_name.lower()}_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(imp, y="feature", x="mean_abs_shap", ax=ax)
    ax.set_title(f"{dataset_name} top 10 SHAP importance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{dataset_name.lower()}_shap_top10.png", dpi=160)
    plt.close(fig)
    save_df(imp, f"{dataset_name.lower()}_shap_top10.csv")
    return imp


def neslin_profit_curve(labeled: pd.DataFrame, best, X_test: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    model, pre, *_ = best
    scores = model.predict_proba(pre.transform(X_test))[:, 1]
    values = X_test.get("Monetary", pd.Series(np.repeat(labeled["Monetary"].mean(), len(X_test)), index=X_test.index)).to_numpy()
    order = np.argsort(-scores)
    rows = []
    for alpha in np.linspace(0, 1, 101):
        n = int(round(len(order) * alpha))
        if n == 0:
            rows.append([alpha, 0, -FIXED_CAMPAIGN_COST_A, -FIXED_CAMPAIGN_COST_A])
            continue
        idx = order[:n]
        beta = scores[idx]
        V = values[idx]
        per_customer = (
            beta * RETENTION_RATE_GAMMA * (V - INCENTIVE_COST_DELTA - CONTACT_COST_KAPPA)
            + beta * (1 - RETENTION_RATE_GAMMA) * (-CONTACT_COST_KAPPA)
            + (1 - beta) * (-INCENTIVE_COST_DELTA - CONTACT_COST_KAPPA)
        )
        profit = float(per_customer.sum() - FIXED_CAMPAIGN_COST_A)
        emp = float(np.maximum(per_customer, 0).sum() - FIXED_CAMPAIGN_COST_A)
        rows.append([alpha, n, profit, emp])
    curve = pd.DataFrame(rows, columns=["target_ratio_alpha", "targeted_customers", "neslin_profit", "emp_approx_profit"])
    best_row = curve.loc[curve["neslin_profit"].idxmax()]
    print(f"Neslin optimum: alpha={best_row['target_ratio_alpha']:.2f}, profit={best_row['neslin_profit']:.2f}", flush=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["target_ratio_alpha"], curve["neslin_profit"], label="Neslin")
    ax.plot(curve["target_ratio_alpha"], curve["emp_approx_profit"], label="EMP approximation")
    ax.axvline(best_row["target_ratio_alpha"], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Target ratio alpha")
    ax.set_ylabel("Profit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{dataset_name.lower()}_neslin_profit_curve.png", dpi=160)
    plt.close(fig)
    save_df(curve, f"{dataset_name.lower()}_neslin_profit_curve.csv")
    return curve


def uplift_demo(labeled: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    demo = labeled[["CustomerID", "churn_90d", "Recency", "Frequency", "Monetary"]].copy()
    demo["treatment"] = rng.binomial(1, 0.5, len(demo))
    base_risk = demo["churn_90d"].to_numpy()
    effect = np.clip(0.18 * (demo["Recency"].to_numpy() > 120) + 0.08 * (demo["Monetary"].rank(pct=True).to_numpy()), 0, 0.35)
    demo["synthetic_retained"] = np.where(
        demo["treatment"].eq(1),
        rng.binomial(1, np.clip(1 - base_risk + effect, 0.02, 0.98)),
        rng.binomial(1, np.clip(1 - base_risk, 0.02, 0.98)),
    )
    X = demo[["Recency", "Frequency", "Monetary"]]
    treated = demo["treatment"].eq(1)
    m_t = RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE, min_samples_leaf=5).fit(X[treated], demo.loc[treated, "synthetic_retained"])
    m_c = RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE, min_samples_leaf=5).fit(X[~treated], demo.loc[~treated, "synthetic_retained"])
    demo["uplift_score"] = m_t.predict_proba(X)[:, 1] - m_c.predict_proba(X)[:, 1]
    ranked = demo.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    qini_rows = []
    for frac in np.linspace(0.05, 1.0, 20):
        head = ranked.head(max(1, int(len(ranked) * frac)))
        t = head["treatment"].eq(1)
        gain = head.loc[t, "synthetic_retained"].mean() - head.loc[~t, "synthetic_retained"].mean()
        qini_rows.append([frac, gain])
    qini = pd.DataFrame(qini_rows, columns=["target_fraction", "synthetic_incremental_retention"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qini["target_fraction"], qini["synthetic_incremental_retention"], marker="o")
    ax.set_title("Synthetic uplift demo only; not a real campaign conclusion")
    ax.set_xlabel("Target fraction")
    ax.set_ylabel("Incremental retention proxy")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "online_synthetic_qini_curve.png", dpi=160)
    plt.close(fig)
    save_df(qini, "online_synthetic_qini_curve.csv")
    return qini


def prepare_telco(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    save_df(profile_telco(df), "telco_column_missing_profile.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    prep_summary = pd.DataFrame(
        [
            ["raw_rows", len(df), ""],
            ["totalcharges_missing_after_numeric", int(df["TotalCharges"].isna().sum()), "blank TotalCharges converted to NA"],
            ["churn_rate", df["Churn"].eq("Yes").mean(), ""],
        ],
        columns=["metric", "value", "note"],
    )
    df = df.dropna(subset=["TotalCharges"]).copy()
    df["churn"] = df["Churn"].eq("Yes").astype(int)
    df = df.drop(columns=["Churn"])
    save_df(prep_summary, "telco_preprocessing_summary.csv")
    return df


def write_results(sections: list[str]) -> None:
    header = "# RESULTS\n\nGenerated by `scripts/run_churn_pipeline.py`.\n\n"
    assumptions = f"""## Assumptions

- Churn inactivity threshold: {CHURN_THRESHOLD_DAYS} days.
- Neslin parameters are explicit scenario assumptions: delta={INCENTIVE_COST_DELTA}, kappa={CONTACT_COST_KAPPA}, gamma={RETENTION_RATE_GAMMA}, fixed A={FIXED_CAMPAIGN_COST_A}.
- Synthetic uplift is a methodology demo only because Online Retail has no observed treatment/control campaign assignment.

"""
    (OUT_DIR / "RESULTS.md").write_text(header + assumptions + "\n".join(sections), encoding="utf-8")


def main() -> None:
    sections: list[str] = []
    online_path, telco_path = ensure_data()

    log_step("1. Raw data profile")
    online_raw = pd.read_excel(online_path)
    telco_raw = pd.read_csv(telco_path)
    online_profile = profile_online(online_raw)
    save_df(online_profile, "online_raw_profile.csv")
    save_df(profile_telco(telco_raw), "telco_raw_profile.csv")
    sections.append("## Raw Data Profile\n\nSee `online_raw_profile.csv` and `telco_raw_profile.csv`.\n")

    log_step("2. Online Retail preprocessing")
    clean = preprocess_online(online_raw)
    sections.append("## Online Retail Preprocessing\n\nSee `online_preprocessing_summary.csv`.\n")

    log_step("3. Online Retail RFM")
    rfm, snapshot = build_rfm(clean)
    sections.append("## RFM Engineering\n\nRFM distributions are saved in `online_rfm_histograms.png`; descriptive stats are in `online_rfm_descriptive_stats.csv`.\n")

    log_step("4. Online Retail dual churn labels")
    labeled = label_online(clean, rfm, snapshot)
    labeled.to_csv(OUT_DIR / "online_customer_rfm_labels.csv", index=False, encoding="utf-8-sig")
    sections.append("## Dual Labeling\n\n90-day inactivity and BG/NBD label agreement is in `online_labeling_summary.csv`.\n")

    log_step("5. Online Retail modeling")
    model_df = labeled.drop(columns=["bgnbd_churn"]).copy()
    online_perf, online_best, online_X_test, online_y_test = evaluate_models(model_df, "churn_90d", "Online", time_col="LastPurchaseDate")
    sections.append("## Online Model Performance\n\nModel metrics are in `online_model_performance.csv`.\n")

    log_step("6. Online Retail SHAP")
    online_shap = shap_plots(online_best, "Online")
    rfm_in_top = online_shap["feature"].str.contains("Recency|Frequency|Monetary", regex=True).sum()
    sections.append(f"## Online SHAP\n\nTop SHAP features are in `online_shap_top10.csv`; {rfm_in_top} of the top 10 features are RFM fields or RFM-derived fields.\n")

    log_step("7. Neslin and EMP profit simulation")
    curve = neslin_profit_curve(labeled, online_best, online_X_test, "Online")
    opt = curve.loc[curve["neslin_profit"].idxmax()]
    sections.append(f"## Neslin Profit Simulation\n\nOptimal target ratio alpha={opt['target_ratio_alpha']:.2f}, expected profit={opt['neslin_profit']:.2f}. Curve saved as `online_neslin_profit_curve.png`.\n")

    log_step("8. Synthetic uplift demo")
    uplift_demo(labeled)
    sections.append("## Synthetic Uplift Demo\n\nQini-style curve saved as `online_synthetic_qini_curve.png`. This is explicitly synthetic and not a real campaign-effect conclusion.\n")

    log_step("9. Harness validation loop demo")
    harness = demo_cases()
    (OUT_DIR / "harness_validation_log.json").write_text(json.dumps(harness, indent=2), encoding="utf-8")
    shutil.copyfile(ROOT / "scripts" / "harness_validation.py", OUT_DIR / "harness_validation.py")
    print(json.dumps(harness, indent=2), flush=True)
    sections.append("## Harness Validation Loop\n\nExecutable code copied to `outputs/harness_validation.py`; sample audit log saved as `harness_validation_log.json`.\n")

    log_step("10. Telco benchmark")
    telco = prepare_telco(telco_path)
    telco_perf, telco_best, _, _ = evaluate_models(telco, "churn", "Telco")
    telco_shap = shap_plots(telco_best, "Telco")
    compare = pd.concat([online_perf.assign(source="Online Retail"), telco_perf.assign(source="Telco")], ignore_index=True)
    save_df(compare, "online_telco_model_performance_comparison.csv")
    shap_compare = pd.concat([online_shap.assign(dataset="Online Retail"), telco_shap.assign(dataset="Telco")], ignore_index=True)
    save_df(shap_compare, "online_telco_shap_top_features_comparison.csv")
    sections.append("## Telco Benchmark\n\nTelco preprocessing/model/SHAP outputs are saved alongside Online Retail comparisons.\n")

    write_results(sections)
    print(f"\nDone. Outputs written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
