"""Re-run CV metrics (same settings as the notebook) and write LaTeX + CSV for the report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soccer_ml import build_model_table, ftr_to_int, load_raw_csvs  # noqa: E402


NUMERIC_FEATURES = [
    "imp_h",
    "imp_d",
    "imp_a",
    "overround",
    "home_gf_roll",
    "home_ga_roll",
    "home_pts_roll",
    "home_home_share_roll",
    "away_gf_roll",
    "away_ga_roll",
    "away_pts_roll",
    "away_home_share_roll",
    "pts_diff_roll",
    "gf_diff_roll",
    "ga_diff_roll",
]
CAT_FEATURES = ["Div"]


def build_pipelines() -> dict[str, Pipeline]:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("prep", preprocess),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "SVM (RBF)": Pipeline(
            steps=[
                ("prep", preprocess),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("prep", preprocess),
                (
                    "clf",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=3,
                        n_estimators=400,
                        max_depth=5,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        min_child_weight=1.0,
                        random_state=42,
                        n_jobs=1,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
    }


def fmt_pm(mean: float, std: float, decimals: int = 3) -> str:
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def main() -> None:
    raw_paths = sorted((ROOT / "data" / "raw").glob("*.csv"))
    if not raw_paths:
        raise SystemExit("No CSVs in data/raw. Run: python scripts/download_data.py")

    matches = load_raw_csvs(raw_paths)
    model_df = build_model_table(matches)

    X = model_df[NUMERIC_FEATURES + CAT_FEATURES]
    y = ftr_to_int(model_df["y_ftr"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "neg_log_loss": "neg_log_loss",
    }

    rows: list[dict[str, object]] = []
    tex_rows: list[str] = []

    for name, est in build_pipelines().items():
        print("Evaluating:", name, flush=True)
        out = cross_validate(est, X, y, cv=cv, scoring=scoring, n_jobs=1)
        acc_m, acc_s = float(np.mean(out["test_accuracy"])), float(np.std(out["test_accuracy"]))
        f1m_m, f1m_s = float(np.mean(out["test_f1_macro"])), float(np.std(out["test_f1_macro"]))
        f1w_m, f1w_s = float(np.mean(out["test_f1_weighted"])), float(np.std(out["test_f1_weighted"]))
        ll_m, ll_s = float(np.mean(-out["test_neg_log_loss"])), float(np.std(-out["test_neg_log_loss"]))

        rows.append(
            {
                "model": name,
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
                "f1_macro_mean": f1m_m,
                "f1_macro_std": f1m_s,
                "f1_weighted_mean": f1w_m,
                "f1_weighted_std": f1w_s,
                "log_loss_mean": ll_m,
                "log_loss_std": ll_s,
            }
        )

        tex_rows.append(
            f"{name} & {fmt_pm(acc_m, acc_s)} & {fmt_pm(f1m_m, f1m_s)} & "
            f"{fmt_pm(f1w_m, f1w_s)} & {fmt_pm(ll_m, ll_s)} \\\\"
        )

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "n_rows": int(len(model_df)),
        "n_splits": 5,
        "random_state": 42,
        "class_balance": model_df["y_ftr"].value_counts(normalize=True).round(4).to_dict(),
    }
    (results_dir / "cv_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(results_dir / "cv_metrics.csv", index=False)

    tex = (
        "% Auto-generated by tools/export_cv_results.py — do not edit by hand.\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Model & Accuracy & Macro-F1 & Weighted-F1 & Log loss \\\\\n"
        "\\midrule\n"
        + "\n".join(tex_rows)
        + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    (ROOT / "report" / "generated_cv_table.tex").write_text(tex, encoding="utf-8")
    print("Wrote results/cv_meta.json, results/cv_metrics.csv, report/generated_cv_table.tex")


if __name__ == "__main__":
    main()
