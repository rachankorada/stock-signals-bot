"""The confidence model: gradient-boosted trees plus isotonic calibration.

Kept in its own module so saved models can be loaded from anywhere.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from backend.features.build import FEATURES, directional
from backend.strategies.library import STRATEGIES

EXTRA = ["stop_atr", "cost_r", "n_agree"]
MODEL_COLS = FEATURES + EXTRA + [f"s_{s}" for s in STRATEGIES]


def model_matrix(ds: pd.DataFrame) -> pd.DataFrame:
    X = directional(ds[FEATURES], ds["direction"].to_numpy())
    for col in EXTRA:
        X[col] = ds[col].to_numpy()
    for s in STRATEGIES:
        X[f"s_{s}"] = (ds["strategy"] == s).astype(int).to_numpy()
    return X[MODEL_COLS].astype("float32")


def calibrator(raw: np.ndarray, won: np.ndarray) -> IsotonicRegression:
    """Maps a raw model score to the win rate such scores actually achieved."""
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, won)


class ConfidenceModel:
    """Estimates the chance that target 1 is reached before the stop."""

    def fit(self, ds: pd.DataFrame, holdback: float = 0.15):
        """Train the trees. `holdback` keeps the newest slice aside for a first calibration;
        pass 0 when a calibration from out-of-sample scores will be set afterwards."""
        ds = ds.sort_values("time")
        cut = int(len(ds) * (1 - holdback))
        fit_part, cal_part = ds.iloc[:cut], ds.iloc[cut:]
        self.gbm = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
            verbose=-1, n_jobs=-1, random_state=7,
        ).fit(model_matrix(fit_part), fit_part["win"])
        self.iso = calibrator(self.raw(cal_part), cal_part["win"]) if len(cal_part) else None
        return self

    def raw(self, ds: pd.DataFrame) -> np.ndarray:
        return self.gbm.predict_proba(model_matrix(ds))[:, 1]

    def predict(self, ds: pd.DataFrame) -> np.ndarray:
        return self.iso.predict(self.raw(ds))
