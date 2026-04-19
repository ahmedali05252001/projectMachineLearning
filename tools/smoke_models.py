import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.soccer_ml import build_model_table, ftr_to_int, load_raw_csvs


def main() -> None:
    paths = sorted((ROOT / "data" / "raw").glob("*.csv"))
    df = build_model_table(load_raw_csvs(paths)).sample(1500, random_state=0)

    num = [
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
    X = df[num + ["Div"]]
    y = ftr_to_int(df["y_ftr"])

    prep = ColumnTransformer(
        [
            ("n", StandardScaler(), num),
            ("c", OneHotEncoder(handle_unknown="ignore"), ["Div"]),
        ]
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    models = {
        "lr": Pipeline(
            [
                ("p", prep),
                ("m", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "svm": Pipeline(
            [
                (
                    "p",
                    prep,
                ),
                ("m", SVC(kernel="rbf", probability=True, class_weight="balanced")),
            ]
        ),
        "xgb": Pipeline(
            [
                ("p", prep),
                (
                    "m",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=3,
                        n_estimators=50,
                        max_depth=3,
                        n_jobs=1,
                        tree_method="hist",
                        random_state=0,
                    ),
                ),
            ]
        ),
    }

    for name, est in models.items():
        out = cross_validate(est, X, y, cv=cv, scoring={"a": "accuracy", "f1": "f1_macro"}, n_jobs=1)
        print(name, float(np.mean(out["test_a"])))


if __name__ == "__main__":
    main()
