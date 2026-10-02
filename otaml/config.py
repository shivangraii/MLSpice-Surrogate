"""Central configuration: paths, parameter ranges, column names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NETLIST_TEMPLATE = ROOT / "netlists" / "ota_template.sp"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

for _d in (DATA_DIR, MODEL_DIR, REPORT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- design-space (Latin Hypercube sampled) -------------------------------
DESIGN_RANGES = {
    "w1_um":    (4.0, 60.0),    # input pair width
    "l1_um":    (0.18, 1.0),    # input pair length
    "w3_um":    (4.0, 60.0),    # PMOS mirror load width
    "l3_um":    (0.18, 1.0),    # PMOS mirror load length
    "ibias_ua": (5.0, 30.0),    # reference current
    "ratio":    (1.0, 3.0),     # tail mirror ratio (tail = ibias * ratio)
    "cl_pf":    (0.5, 2.0),     # load capacitor
    "vdd":      (1.6, 2.0),     # supply
    "temp":     (-40.0, 125.0), # temperature in C
}
LOG_SCALE_COLS = ["w1_um", "l1_um", "w3_um", "l3_um", "ibias_ua", "cl_pf"]

# ---- process variation (Monte-Carlo, gaussian) -----------------------------
PROCESS_SIGMA = {
    "dvtn": 0.02,   # NMOS Vth shift in V
    "dvtp": 0.02,   # PMOS Vth shift in V
    "dkpn": 0.05,   # NMOS kp relative shift
    "dkpp": 0.05,   # PMOS kp relative shift
}
PROCESS_CLIP_SIGMA = 3.0

DESIGN_COLS = list(DESIGN_RANGES) + list(PROCESS_SIGMA)

# ---- nominal device numbers (Level-1 models) -------------------------------
VTN0, VTP0 = 0.50, 0.50
KPN0, KPP0 = 170e-6, 60e-6
LAMBDA_N, LAMBDA_P = 0.02, 0.03       # lambda = LAMBDA / L[um]
W_REF_UM, L_REF_UM = 10.0, 1.0        # reference diode M0 (tail W = W_REF*ratio)
VCM_FRAC = 0.70                       # input common mode = VCM_FRAC * vdd

# ---- surrogate targets -----------------------------------------------------
TARGET_COLS = ["gain_db", "log_ugbw", "pm_deg", "power_uw", "vout", "vtail"]
MEASURED_COLS = TARGET_COLS + ["vd1"]
RESIDUAL_COLS = [f"res_{c}" for c in TARGET_COLS]

# ---- validity filter for *healthy* samples ---------------------------------
MIN_GAIN_DB = 25.0
MIN_VTAIL = 0.12          # keeps the tail mirror in saturation
VOUT_MARGIN = 0.20        # output must sit this far from both rails

# ---- fault classes ---------------------------------------------------------
FAULT_CLASSES = [
    "healthy",
    "m1_ds_short",     # drain-source short on input transistor M1
    "m4_gate_open",    # mirror transistor M4 gate floating / pulled to VDD
    "bias_low",        # bias current drifted low
    "vth_drift",       # input-pair threshold drift (aging)
    "cap_shift",       # load capacitor value shift
]
