"""Train a calibrated XGBoost classifier, optimized for PR-AUC on an imbalanced target."""
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

FEATURE_COLS = ["cvss_base_score", "epss_score"]  # extend with encoded categoricals + NLP embeddings
TARGET_COL = "is_kev"


def train(df: pd.DataFrame, model_path: str = "model.joblib") -> XGBClassifier:
    X = df[FEATURE_COLS].fillna(-1)
    y = df[TARGET_COL].astype(int)

    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),  # counter the <5% positive-class imbalance
        eval_metric="aucpr",
        random_state=42,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(base_model, X, y, cv=skf, scoring="average_precision")
    print(f"5-fold PR-AUC: {scores.mean():.4f} +/- {scores.std():.4f}")

    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=skf)
    calibrated.fit(X, y)
    joblib.dump(calibrated, model_path)
    return calibrated


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/features.parquet")
    train(df)
