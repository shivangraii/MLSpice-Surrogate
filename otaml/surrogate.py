"""Surrogate model wrapper: design params -> (gain, UGBW, PM, power, Vout, Vtail)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import design_matrix

SURROGATE_PATH = C.MODEL_DIR / "surrogate.joblib"


class Surrogate:
    def __init__(self, hidden=(128, 128, 64), seed=0, max_iter=600):
        self.hidden, self.seed, self.max_iter = hidden, seed, max_iter
        self.xs, self.ys = StandardScaler(), StandardScaler()
        self.nets: list[MLPRegressor] = []

    def fit(self, df: pd.DataFrame, n_ensemble: int = 3):
        X = self.xs.fit_transform(design_matrix(df))
        Y = self.ys.fit_transform(df[C.TARGET_COLS].to_numpy(float))
        self.nets = []
        for k in range(n_ensemble):  # small ensemble = smoother, more accurate predictions
            net = MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu", solver="adam", learning_rate_init=2e-3,
                               max_iter=self.max_iter, early_stopping=True, validation_fraction=0.1,
                               n_iter_no_change=30, random_state=self.seed + k)
            net.fit(X, Y)
            self.nets.append(net)
        return self

    def predict_array(self, X_raw: np.ndarray) -> np.ndarray:
        X = self.xs.transform(X_raw)
        Y = np.mean([n.predict(X) for n in self.nets], axis=0)
        return self.ys.inverse_transform(Y)

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        Y = self.predict_array(design_matrix(df))
        return pd.DataFrame(Y, columns=C.TARGET_COLS, index=df.index)

    def predict_physical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same as predict_df but with UGBW in Hz instead of log10(Hz)."""
        out = self.predict_df(df)
        out["ugbw_hz"] = 10 ** out["log_ugbw"]
        return out

    def save(self, path=SURROGATE_PATH):
        joblib.dump(self, path)

    @staticmethod
    def load(path=SURROGATE_PATH) -> "Surrogate":
        return joblib.load(path)
