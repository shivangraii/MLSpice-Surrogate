"""Streamlit demo.  Run:  streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd
import streamlit as st

from otaml import config as C
from otaml.classifier import FaultDiagnoser, add_measurement_noise
from otaml.faults import sample_fault
from otaml.optimize import BOUNDS, FIXED, VARS, summarize, surrogate_cost
from otaml.simulate import find_ngspice, simulate
from otaml.surrogate import Surrogate

st.set_page_config(page_title="OTA Surrogate + Fault Diagnosis", layout="wide")


@st.cache_resource
def load_models():
    return Surrogate.load(), FaultDiagnoser.load()


try:
    sur, diag = load_models()
except Exception as e:  # models missing
    st.error(f"Models not found ({e}). Run `bash run_all.sh` first.")
    st.stop()

try:
    find_ngspice(); HAVE_SPICE = True
except FileNotFoundError:
    HAVE_SPICE = False
    st.warning("ngspice not found: surrogate works, but 'verify' / 'inject fault' buttons are disabled.")

st.title("5T OTA: ML surrogate, fault diagnosis and auto-sizing")
st.caption("ngspice generates the data; the surrogate replaces it for fast design exploration.")

# ---------------- sidebar: design parameters -------------------------------
sb = st.sidebar
sb.header("Design parameters")
d = {}
for k, (lo, hi) in C.DESIGN_RANGES.items():
    default = float(np.sqrt(lo * hi)) if k in C.LOG_SCALE_COLS else float((lo + hi) / 2)
    if k == "temp":
        default = 27.0
    if k == "vdd":
        default = 1.8
    d[k] = sb.slider(k, float(lo), float(hi), default)
sb.header("Process variation")
for k, sig in C.PROCESS_SIGMA.items():
    d[k] = sb.slider(k, -3 * sig, 3 * sig, 0.0, step=sig / 10)
design_df = pd.DataFrame([d])

tab1, tab2, tab3, tab4 = st.tabs(["Surrogate", "Fault diagnosis", "Auto-sizing", "Results"])

# ---------------- tab 1 ----------------------------------------------------
with tab1:
    p = sur.predict_physical(design_df).iloc[0]
    c = st.columns(4)
    c[0].metric("Gain (dB)", f"{p.gain_db:.2f}")
    c[1].metric("UGBW (MHz)", f"{p.ugbw_hz / 1e6:.1f}")
    c[2].metric("Phase margin (deg)", f"{p.pm_deg:.1f}")
    c[3].metric("Power (uW)", f"{p.power_uw:.1f}")
    st.caption(f"Predicted Vout = {p.vout:.3f} V, Vtail = {p.vtail:.3f} V  (instant, no SPICE run)")
    if st.button("Verify with ngspice", disabled=not HAVE_SPICE):
        r = simulate(d)
        if r["ok"] and not np.isnan(r["ugbw_hz"]):
            tab = pd.DataFrame({
                "metric": ["gain_db", "ugbw_mhz", "pm_deg", "power_uw"],
                "ngspice": [r["gain_db"], r["ugbw_hz"] / 1e6, r["pm_deg"], r["power_uw"]],
                "surrogate": [p.gain_db, p.ugbw_hz / 1e6, p.pm_deg, p.power_uw]})
            tab["error_%"] = (tab.surrogate / tab.ngspice - 1).abs() * 100
            st.dataframe(tab.round(3), hide_index=True)
        else:
            st.warning("This design does not give a valid amplifier in ngspice (no unity-gain crossing).")

# ---------------- tab 2 ----------------------------------------------------
with tab2:
    st.write("Inject a fault into the circuit defined in the sidebar, simulate it, and let the classifier diagnose it.")
    a, b, c3 = st.columns(3)
    fault_name = a.selectbox("Fault to inject", C.FAULT_CLASSES, index=1)
    seed = int(b.number_input("Severity seed", 0, 10_000, 1))
    noise = c3.slider("Measurement noise (relative sigma)", 0.0, 0.05, 0.01, 0.005)
    if st.button("Inject fault, simulate and diagnose", disabled=not HAVE_SPICE):
        f = sample_fault(fault_name, np.random.default_rng(seed))
        r = simulate(d, f)
        row = pd.DataFrame([{**d, **r}])
        row = add_measurement_noise(row, noise, seed=seed)
        proba = diag.predict_proba(row).iloc[0].sort_values(ascending=False)
        st.subheader(f"Diagnosis: **{proba.index[0]}**  ({proba.iloc[0] * 100:.1f}% confidence)")
        st.bar_chart(proba)
        pe = sur.predict_physical(design_df).iloc[0]
        cmp = pd.DataFrame({"measured": [r["gain_db"], r["pm_deg"], r["power_uw"], r["vout"], r["vtail"]],
                            "expected if healthy": [pe.gain_db, pe.pm_deg, pe.power_uw, pe.vout, pe.vtail]},
                           index=["gain_db", "pm_deg", "power_uw", "vout", "vtail"])
        st.write("Measured vs surrogate-expected (the residuals drive the diagnosis):")
        st.dataframe(cmp.round(3))
        st.caption(f"Injected fault details: {f}")

# ---------------- tab 3 ----------------------------------------------------
with tab3:
    st.write("Minimise power subject to specs. Surrogate-based differential evolution, then one ngspice check. (Nominal PVT, CL = 1 pF)")
    a, b, c3 = st.columns(3)
    g = a.number_input("Min gain (dB)", 30.0, 60.0, 45.0)
    u = b.number_input("Min UGBW (MHz)", 10.0, 300.0, 80.0)
    m = c3.number_input("Min PM (deg)", 30.0, 90.0, 60.0)
    if st.button("Find a design"):
        from scipy.optimize import differential_evolution
        spec = dict(gain=g, ugbw=u * 1e6, pm=m)
        res = differential_evolution(lambda X: surrogate_cost(X, sur, spec, 0.03), BOUNDS, maxiter=300, popsize=40,
                                     seed=0, polish=False, tol=0, vectorized=True, updating="deferred")
        st.write("Optimiser used the surrogate only (about 12,000 evaluations).")
        st.json({k: round(float(v), 3) for k, v in zip(VARS, res.x)})
        if HAVE_SPICE:
            out = summarize("surrogate DE", res.x, 1, 0, spec)
            st.success("Specs met in ngspice" if out["specs_met"] else "Specs NOT met in ngspice (try tighter guard band)")
            st.write({k: out[k] for k in ("power_uw", "gain_db", "ugbw_mhz", "pm_deg")})

# ---------------- tab 4 ----------------------------------------------------
with tab4:
    for name, fn in [("Surrogate vs ngspice", "surrogate_pred_vs_actual.png"),
                     ("Speed benchmark", "benchmark_speed.png"),
                     ("Fault confusion matrix", "fault_confusion_matrix.png"),
                     ("Feature ablation", "fault_ablation.png"),
                     ("Noise robustness", "fault_noise_robustness.png")]:
        p_ = C.FIG_DIR / fn
        if p_.exists():
            st.subheader(name); st.image(str(p_))
    bp = C.REPORT_DIR / "benchmark.json"
    if bp.exists():
        st.subheader("Benchmark numbers"); st.json(json.loads(bp.read_text()))
