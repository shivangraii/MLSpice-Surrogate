"""Surrogate-assisted analog sizing.

Minimise power subject to gain / UGBW / phase-margin specs at nominal PVT.
Compares  (a) random search on ngspice, (b) differential evolution on ngspice,
          (c) differential evolution on the surrogate + ONE ngspice verification.

    python -m otaml.optimize --gain 45 --ugbw 80e6 --pm 60
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from . import config as C
from .features import clean_measurements
from .simulate import simulate
from .surrogate import Surrogate

VARS = ["w1_um", "l1_um", "w3_um", "l3_um", "ibias_ua", "ratio"]
BOUNDS = [C.DESIGN_RANGES[k] for k in VARS]
FIXED = dict(cl_pf=1.0, vdd=1.8, temp=27.0, dvtn=0.0, dvtp=0.0, dkpn=0.0, dkpp=0.0)


def to_design(x) -> dict:
    d = dict(zip(VARS, map(float, x)))
    d.update(FIXED)
    return d


def violation(gain, ugbw, pm, vtail, vout, vdd, spec) -> np.ndarray:
    """Sum of relative spec violations (0 = all specs met)."""
    v = (np.maximum(0, (spec["gain"] - gain) / spec["gain"])
         + np.maximum(0, (spec["ugbw"] - ugbw) / spec["ugbw"])
         + np.maximum(0, (spec["pm"] - pm) / spec["pm"])
         + np.maximum(0, (C.MIN_VTAIL - vtail) / C.MIN_VTAIL)
         + np.maximum(0, (C.VOUT_MARGIN - vout) / C.VOUT_MARGIN)
         + np.maximum(0, (vout - (vdd - C.VOUT_MARGIN)) / C.VOUT_MARGIN))
    return v


def surrogate_cost(X: np.ndarray, sur: Surrogate, spec: dict, guard: float) -> np.ndarray:
    df = pd.DataFrame(X.T, columns=VARS)
    for k, v in FIXED.items():
        df[k] = v
    p = sur.predict_physical(df)
    g = dict(gain=spec["gain"] * (1 + guard), ugbw=spec["ugbw"] * (1 + guard), pm=spec["pm"])
    viol = violation(p["gain_db"].to_numpy(), p["ugbw_hz"].to_numpy(), p["pm_deg"].to_numpy(),
                     p["vtail"].to_numpy(), p["vout"].to_numpy(), FIXED["vdd"], g)
    return p["power_uw"].to_numpy() + 2000.0 * viol


def spice_eval(x, spec) -> tuple[float, dict]:
    r = simulate(to_design(x))
    if not r["ok"] or np.isnan(r["ugbw_hz"]) or np.isnan(r["pm_deg"]):
        return 1e6, r
    viol = float(violation(r["gain_db"], r["ugbw_hz"], r["pm_deg"], r["vtail"], r["vout"], FIXED["vdd"], spec))
    return r["power_uw"] + 2000.0 * viol, r


def summarize(name, x, calls, secs, spec) -> dict:
    r = simulate(to_design(x))
    met = bool(violation(r["gain_db"], r["ugbw_hz"], r["pm_deg"], r["vtail"], r["vout"], FIXED["vdd"], spec) == 0)
    return dict(method=name, spice_calls=calls, seconds=round(secs, 2), specs_met=met,
                power_uw=round(r["power_uw"], 2), gain_db=round(r["gain_db"], 2),
                ugbw_mhz=round(r["ugbw_hz"] / 1e6, 2), pm_deg=round(r["pm_deg"], 2),
                design={k: round(float(v), 4) for k, v in zip(VARS, x)})


def run(spec: dict, budget: int = 600, seed: int = 0, guard: float = 0.03) -> list[dict]:
    sur = Surrogate.load()
    rows = []

    # (a) random search directly on ngspice -----------------------------------
    rng = np.random.default_rng(seed)
    t = time.perf_counter()
    best_x, best_c = None, np.inf
    for _ in range(budget):
        x = [rng.uniform(lo, hi) if k not in C.LOG_SCALE_COLS else np.exp(rng.uniform(np.log(lo), np.log(hi)))
             for k, (lo, hi) in zip(VARS, BOUNDS)]
        c, _ = spice_eval(x, spec)
        if c < best_c:
            best_x, best_c = x, c
    rows.append(summarize("random search (ngspice)", best_x, budget, time.perf_counter() - t, spec))

    # (b) DE directly on ngspice -----------------------------------------------
    t = time.perf_counter(); n_calls = [0]

    def f(x):
        n_calls[0] += 1
        return spice_eval(x, spec)[0]
    iters = max(1, budget // (10 * len(VARS)) - 1)
    res = differential_evolution(f, BOUNDS, maxiter=iters, popsize=10, seed=seed, polish=False, tol=0)
    rows.append(summarize("differential evolution (ngspice)", res.x, n_calls[0], time.perf_counter() - t, spec))

    # (c) DE on the surrogate, verified once in ngspice ------------------------
    t = time.perf_counter()
    res = differential_evolution(lambda X: surrogate_cost(X, sur, spec, guard), BOUNDS, maxiter=300, popsize=40,
                                 seed=seed, polish=False, tol=0, vectorized=True, updating="deferred")
    rows.append(summarize("surrogate DE + 1 ngspice check", res.x, 1, time.perf_counter() - t, spec))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, default=45.0, help="min DC gain (dB)")
    ap.add_argument("--ugbw", type=float, default=80e6, help="min unity-gain bandwidth (Hz)")
    ap.add_argument("--pm", type=float, default=60.0, help="min phase margin (deg)")
    ap.add_argument("--budget", type=int, default=600, help="ngspice calls for the ngspice-only baselines")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    spec = dict(gain=args.gain, ugbw=args.ugbw, pm=args.pm)
    rows = run(spec, args.budget, args.seed)
    tab = pd.DataFrame(rows).drop(columns="design")
    print(f"\nSpecs: gain>={args.gain} dB, UGBW>={args.ugbw/1e6:g} MHz, PM>={args.pm} deg   (minimise power)\n")
    print(tab.to_string(index=False))
    (C.REPORT_DIR / "optimization.json").write_text(json.dumps(dict(spec=spec, results=rows), indent=2))


if __name__ == "__main__":
    main()
