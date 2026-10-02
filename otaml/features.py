"""Feature engineering shared by training, benchmarking and the app."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Design/process params -> model input (log for sizes, currents and cap)."""
    X = df[C.DESIGN_COLS].copy()
    for c in C.LOG_SCALE_COLS:
        X[c] = np.log(X[c])
    return X.to_numpy(dtype=float)


def clean_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-friendly columns. Faulty circuits can lose the unity-gain crossing:
    those get floor values (log_ugbw=3 -> 1 kHz, pm=0) plus an explicit flag."""
    out = df.copy()
    out["no_ugbw"] = out["ugbw_hz"].isna().astype(int)
    out["log_ugbw"] = np.log10(out["ugbw_hz"].clip(lower=1e3)).fillna(3.0)
    out["pm_deg"] = out["pm_deg"].fillna(0.0)
    out["gain_db"] = out["gain_db"].clip(-60.0, 100.0)   # shorted circuits give absurd dB values
    for c in ("gain_db", "power_uw", "vout", "vtail", "vd1"):
        out[c] = out[c].fillna(out[c].median() if out[c].notna().any() else 0.0)
    return out


def valid_healthy(df: pd.DataFrame) -> pd.Series:
    """Mask of healthy samples that behave like a real amplifier (in saturation, decent gain)."""
    return (
        (df["ok"] == 1)
        & df["gain_db"].ge(C.MIN_GAIN_DB)
        & df["ugbw_hz"].notna()
        & df["pm_deg"].notna()
        & df["vtail"].ge(C.MIN_VTAIL)
        & df["vout"].between(C.VOUT_MARGIN, df["vdd"] - C.VOUT_MARGIN)
    )


def residual_features(df: pd.DataFrame, surrogate) -> pd.DataFrame:
    """Measured minus surrogate-expected (healthy) values for every surrogate target."""
    pred = surrogate.predict_df(df)
    res = pd.DataFrame(index=df.index)
    for c in C.TARGET_COLS:
        res[f"res_{c}"] = df[c].to_numpy() - pred[c].to_numpy()
    return res
