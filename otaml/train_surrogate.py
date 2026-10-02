"""Train the surrogate and compare against baselines.

    python -m otaml.train_surrogate
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import clean_measurements, design_matrix, valid_healthy
from .surrogate import Surrogate


def mape(y, p):
    return float(np.mean(np.abs((y - p) / np.maximum(np.abs(y), 1e-9))) * 100)


def score(Y_true, Y_pred):
    out = {}
    for j, c in enumerate(C.TARGET_COLS):
        out[c] = dict(r2=float(r2_score(Y_true[:, j], Y_pred[:, j])),
                      mape=mape(Y_true[:, j], Y_pred[:, j]),
                      mae=float(np.mean(np.abs(Y_true[:, j] - Y_pred[:, j]))))
    return out


def main():
    raw = pd.read_csv(C.DATA_DIR / "healthy.csv")
    mask = valid_healthy(raw)
    print(f"valid healthy samples: {mask.sum()} / {len(raw)}")
    df = clean_measurements(raw[mask].reset_index(drop=True))
    tr, te = train_test_split(df, test_size=0.2, random_state=0)
    Yte = te[C.TARGET_COLS].to_numpy(float)

    results = {}
    # --- baselines ---------------------------------------------------------
    xs, ys = StandardScaler(), StandardScaler()
    Xtr = xs.fit_transform(design_matrix(tr)); Ytr = ys.fit_transform(tr[C.TARGET_COLS].to_numpy(float))
    Xte = xs.transform(design_matrix(te))
    lin = Ridge(alpha=1.0).fit(Xtr, Ytr)
    results["linear_ridge"] = score(Yte, ys.inverse_transform(lin.predict(Xte)))
    rf = RandomForestRegressor(300, n_jobs=-1, random_state=0).fit(Xtr, Ytr)
    results["random_forest"] = score(Yte, ys.inverse_transform(rf.predict(Xte)))
    # --- MLP surrogate -----------------------------------------------------
    sur = Surrogate().fit(tr)
    pred = sur.predict_df(te).to_numpy()
    results["mlp_ensemble"] = score(Yte, pred)
    sur.save()

    summary = pd.DataFrame({m: {c: results[m][c]["mape"] for c in C.TARGET_COLS} for m in results}).T
    r2 = pd.DataFrame({m: {c: results[m][c]["r2"] for c in C.TARGET_COLS} for m in results}).T
    print("\nTest MAPE (%)\n", summary.round(2).to_string())
    print("\nTest R2\n", r2.round(4).to_string())
    (C.REPORT_DIR / "surrogate_metrics.json").write_text(json.dumps(
        dict(n_train=len(tr), n_test=len(te), metrics=results), indent=2))

    # --- plots: predicted vs actual ----------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    labels = dict(gain_db="Gain (dB)", log_ugbw="log10 UGBW (Hz)", pm_deg="Phase margin (deg)",
                  power_uw="Power (uW)", vout="Vout DC (V)", vtail="Vtail (V)")
    for j, (ax, c) in enumerate(zip(axes.ravel(), C.TARGET_COLS)):
        ax.scatter(Yte[:, j], pred[:, j], s=6, alpha=0.5)
        lo, hi = Yte[:, j].min(), Yte[:, j].max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"{labels[c]}  R2={results['mlp_ensemble'][c]['r2']:.3f}")
        ax.set_xlabel("ngspice"); ax.set_ylabel("surrogate")
    fig.suptitle("Surrogate vs ngspice (held-out test set)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "surrogate_pred_vs_actual.png", dpi=140); plt.close(fig)

    ax = summary.T.plot(kind="bar", figsize=(8, 4), logy=True)
    ax.set_ylabel("MAPE % (log)"); ax.set_title("Model comparison (lower is better)")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "surrogate_model_comparison.png", dpi=140); plt.close()
    print("saved models/surrogate.joblib and reports/")


if __name__ == "__main__":
    main()
