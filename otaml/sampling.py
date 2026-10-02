"""Latin-Hypercube design sampling + Monte-Carlo process variation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from . import config as C


def sample_designs(n: int, seed: int = 0, process_variation: bool = True) -> pd.DataFrame:
    """n designs: LHS over DESIGN_RANGES (log-uniform for sizes/currents) + gaussian process shifts."""
    keys = list(C.DESIGN_RANGES)
    u = qmc.LatinHypercube(d=len(keys), seed=seed).random(n)
    cols = {}
    for j, k in enumerate(keys):
        lo, hi = C.DESIGN_RANGES[k]
        if k in C.LOG_SCALE_COLS:
            cols[k] = np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo)))
        else:
            cols[k] = lo + u[:, j] * (hi - lo)
    rng = np.random.default_rng(seed + 10_000)
    for k, sig in C.PROCESS_SIGMA.items():
        if process_variation:
            cols[k] = np.clip(rng.normal(0.0, sig, n), -C.PROCESS_CLIP_SIGMA * sig, C.PROCESS_CLIP_SIGMA * sig)
        else:
            cols[k] = np.zeros(n)
    return pd.DataFrame(cols)[C.DESIGN_COLS]
