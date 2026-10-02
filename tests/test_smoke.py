import math

import numpy as np
import pytest

from otaml import config as C
from otaml.faults import sample_fault
from otaml.features import clean_measurements, design_matrix
from otaml.netlist import build_netlist
from otaml.sampling import sample_designs
from otaml.simulate import find_ngspice, simulate

try:
    find_ngspice(); HAVE_SPICE = True
except FileNotFoundError:
    HAVE_SPICE = False

needs_spice = pytest.mark.skipif(not HAVE_SPICE, reason="ngspice not installed")


def test_sampling_in_range():
    df = sample_designs(200, seed=1)
    for k, (lo, hi) in C.DESIGN_RANGES.items():
        assert df[k].between(lo, hi).all()
    assert list(df.columns) == C.DESIGN_COLS
    assert design_matrix(df).shape == (200, len(C.DESIGN_COLS))


def test_netlist_has_no_unfilled_placeholders():
    d = sample_designs(1, seed=2).iloc[0].to_dict()
    net = build_netlist(d, sample_fault("m1_ds_short", np.random.default_rng(0)))
    body = net.split(".control")[0]
    assert "$" not in body
    assert "RFSH" in body


@needs_spice
def test_healthy_circuit_is_a_sane_amplifier():
    d = sample_designs(1, seed=3).iloc[0].to_dict()
    r = simulate(d)
    assert r["ok"] == 1
    assert 20 < r["gain_db"] < 80
    assert r["ugbw_hz"] > 1e6
    assert 30 < r["pm_deg"] < 100
    assert r["power_uw"] > 0


@needs_spice
def test_faults_change_the_measurements():
    d = sample_designs(1, seed=4).iloc[0].to_dict()
    base = simulate(d)
    rng = np.random.default_rng(0)
    bias = simulate(d, sample_fault("bias_low", rng))
    assert bias["power_uw"] < 0.8 * base["power_uw"]
    cap = simulate(d, sample_fault("cap_shift", rng))
    assert cap["ugbw_hz"] < 0.5 * base["ugbw_hz"]
    gate = simulate(d, sample_fault("m4_gate_open", rng))
    assert math.isnan(gate["ugbw_hz"]) or gate["gain_db"] < base["gain_db"] - 15


def test_clean_measurements_handles_nan():
    import pandas as pd
    df = pd.DataFrame(dict(gain_db=[-300.0, 40.0], ugbw_hz=[np.nan, 1e7], pm_deg=[np.nan, 70.0],
                           power_uw=[10.0, 20.0], vout=[1.0, 1.1], vtail=[0.3, 0.4], vd1=[1.0, 1.1]))
    out = clean_measurements(df)
    assert out["no_ugbw"].tolist() == [1, 0]
    assert out["gain_db"].min() >= -60
    assert not out[["log_ugbw", "pm_deg"]].isna().any().any()
