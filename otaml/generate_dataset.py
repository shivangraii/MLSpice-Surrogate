"""Generate the ngspice datasets.

    python -m otaml.generate_dataset --n-healthy 6000 --n-per-fault 800

Outputs
    data/healthy.csv  -> surrogate training data (healthy circuits, PVT + Monte-Carlo sweep)
    data/faults.csv   -> fault-diagnosis data (healthy + 5 fault classes, different seed than healthy.csv)
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from . import config as C
from .faults import sample_fault
from .sampling import sample_designs
from .simulate import find_ngspice, simulate


def _job(args):
    design, fault_name, seed = args
    rng = np.random.default_rng(seed)
    fault = sample_fault(fault_name, rng)
    res = simulate(design, fault)
    return {**design, "fault": fault_name, "severity": fault["severity"], **res}


def run_jobs(jobs, workers: int) -> pd.DataFrame:
    if workers <= 1:
        rows = [_job(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            rows = pool.map(_job, jobs, chunksize=32)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-healthy", type=int, default=6000, help="healthy samples for the surrogate")
    ap.add_argument("--n-per-fault", type=int, default=800, help="samples per class for fault dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()
    find_ngspice()

    t0 = time.time()
    # ---- 1) healthy dataset for the surrogate ------------------------------
    designs = sample_designs(args.n_healthy, seed=args.seed).to_dict("records")
    jobs = [(d, "healthy", args.seed + i) for i, d in enumerate(designs)]
    healthy = run_jobs(jobs, args.workers)
    healthy.to_csv(C.DATA_DIR / "healthy.csv", index=False)
    print(f"[healthy] {len(healthy)} sims in {time.time() - t0:.1f}s -> data/healthy.csv")

    # ---- 2) fault dataset (independent designs, so no leakage) -------------
    t1 = time.time()
    jobs = []
    for ci, name in enumerate(C.FAULT_CLASSES):
        ds = sample_designs(args.n_per_fault, seed=args.seed + 1000 + ci).to_dict("records")
        jobs += [(d, name, args.seed + 50_000 + ci * 100_000 + i) for i, d in enumerate(ds)]
    faults = run_jobs(jobs, args.workers)
    faults.to_csv(C.DATA_DIR / "faults.csv", index=False)
    print(f"[faults ] {len(faults)} sims in {time.time() - t1:.1f}s -> data/faults.csv")
    print(faults["fault"].value_counts().to_string())


if __name__ == "__main__":
    main()
