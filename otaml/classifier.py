"""Fault classifier helpers (feature sets, inference wrapper)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import clean_measurements, residual_features

CLF_PATH = C.MODEL_DIR / "fault_clf.joblib"

FEATURE_SETS = {
    "measured_only":       lambda m, d, r: m,
    "measured+design":     lambda m, d, r: pd.concat([m, d], axis=1),
    "measured+residuals":  lambda m, d, r: pd.concat([m, r], axis=1),
}


def add_measurement_noise(df: pd.DataFrame, level: float, seed: int = 0) -> pd.DataFrame:
    """Emulate instrument noise. `level` is a relative sigma (0.01 = 1 %).
    Gain gets additive dB noise (level * 50 dB), the rest multiplicative noise."""
    if level <= 0:
        return df.copy()
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    out["gain_db"] = out["gain_db"] + rng.normal(0, level * 50.0, n)
    for c in ("ugbw_hz", "pm_deg", "power_uw", "vout", "vtail", "vd1"):
        out[c] = out[c] * (1 + rng.normal(0, level, n))
    return out


def build_features(df: pd.DataFrame, surrogate, kind: str = "measured+residuals") -> pd.DataFrame:
    df = clean_measurements(df)
    meas = df[C.MEASURED_COLS + ["no_ugbw"]]
    design = df[C.DESIGN_COLS]
    res = residual_features(df, surrogate)
    return FEATURE_SETS[kind](meas, design, res)


class FaultDiagnoser:
    """Bundles surrogate + classifier: raw measurements + design intent -> fault probabilities."""

    def __init__(self, surrogate, clf, kind="measured+residuals"):
        self.surrogate, self.clf, self.kind = surrogate, clf, kind

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        X = build_features(df, self.surrogate, self.kind)
        P = self.clf.predict_proba(X)
        return pd.DataFrame(P, columns=self.clf.classes_, index=df.index)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(df).idxmax(axis=1).to_numpy()

    def save(self, path=CLF_PATH):
        joblib.dump(self, path)

    @staticmethod
    def load(path=CLF_PATH) -> "FaultDiagnoser":
        return joblib.load(path)
