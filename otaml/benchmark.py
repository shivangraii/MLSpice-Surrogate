"""Speed + accuracy benchmark: ngspice vs surrogate on fresh, unseen designs.

    python -m otaml.benchmark --n 1000
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .features import clean_measurements, design_matrix, valid_healthy
from .sampling import sample_designs
from .simulate import simulate
from .surrogate import Surrogate


def _sim(design):
    return simulate(design)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    sur = Surrogate.load()
    designs = sample_designs(args.n, seed=777_777)          # unseen seed
    recs = designs.to_dict("records")

    t = time.perf_counter(); rows = [_sim(d) for d in recs]; t_spice_seq = time.perf_counter() - t
    t = time.perf_counter()
    with Pool(args.workers) as p:
        p.map(_sim, recs, chunksize=16)
    t_spice_par = time.perf_counter() - t

    X = design_matrix(designs)
    sur.predict_array(X[:2])                                   # warm-up
    t = time.perf_counter(); sur.predict_array(X); t_batch = time.perf_counter() - t
    t = time.perf_counter()
    for i in range(args.n):
        sur.predict_array(X[i:i + 1])
    t_single = time.perf_counter() - t

    truth = pd.DataFrame(rows)
    ok = valid_healthy(pd.concat([designs, truth], axis=1))
    truth = clean_measurements(truth[ok])
    pred = sur.predict_df(designs[ok.to_numpy()])
    err = {c: float(np.mean(np.abs((truth[c].to_numpy() - pred[c].to_numpy()) /
                                   np.maximum(np.abs(truth[c].to_numpy()), 1e-9))) * 100) for c in C.TARGET_COLS}

    out = dict(
        n=args.n, workers=args.workers,
        ngspice_sequential_s=t_spice_seq, ngspice_parallel_s=t_spice_par,
        surrogate_batch_s=t_batch, surrogate_single_calls_s=t_single,
        speedup_batch_vs_seq=t_spice_seq / t_batch,
        speedup_single_vs_seq=t_spice_seq / t_single,
        speedup_batch_vs_parallel=t_spice_par / t_batch,
        mape_percent=err,
    )
    (C.REPORT_DIR / "benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    labels = ["ngspice\n(1 core)", f"ngspice\n({args.workers} cores)", "surrogate\n(1 call each)", "surrogate\n(batch)"]
    vals = [t_spice_seq, t_spice_par, t_single, t_batch]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color=["#c0504d", "#e6a09e", "#4f81bd", "#17375e"])
    ax.set_yscale("log"); ax.set_ylabel(f"seconds for {args.n} evaluations (log)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3g}s", ha="center", va="bottom")
    ax.set_title("Evaluation time: SPICE vs surrogate")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "benchmark_speed.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
