"""Component-level fault models. Each fault is a small modification of the netlist."""
from __future__ import annotations

import numpy as np

from .config import FAULT_CLASSES


def sample_fault(name: str, rng: np.random.Generator) -> dict:
    """Return a fault specification (what to change in the netlist) with a random severity."""
    spec = dict(name=name, severity=0.0, ibias_scale=1.0, dvt_in=0.0,
                cl_scale=1.0, m4gate="d1", fault_lines="")
    if name == "healthy":
        pass
    elif name == "m1_ds_short":
        r = float(np.exp(rng.uniform(np.log(20.0), np.log(2000.0))))   # 20 ohm .. 2 kohm
        spec["severity"] = r
        spec["fault_lines"] = f"RFSH d1 tail {r:.4g}"
    elif name == "m4_gate_open":
        r = float(np.exp(rng.uniform(np.log(200.0), np.log(20e3))))    # pull-up strength
        spec["severity"] = r
        spec["m4gate"] = "g4f"
        spec["fault_lines"] = f"RG4F g4f vdd {r:.4g}"
    elif name == "bias_low":
        s = float(rng.uniform(0.25, 0.65))
        spec["severity"] = s
        spec["ibias_scale"] = s
    elif name == "vth_drift":
        d = float(rng.uniform(0.10, 0.25))
        spec["severity"] = d
        spec["dvt_in"] = d
    elif name == "cap_shift":
        c = float(rng.uniform(3.0, 8.0))
        spec["severity"] = c
        spec["cl_scale"] = c
    else:
        raise ValueError(f"unknown fault {name!r}; choose from {FAULT_CLASSES}")
    return spec
