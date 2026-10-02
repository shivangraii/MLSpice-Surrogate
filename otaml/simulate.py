"""Run ngspice in batch mode and parse the measurements."""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .netlist import build_netlist

_KV = re.compile(r"(\w+)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def find_ngspice() -> str:
    """Locate the ngspice binary. Override with env var NGSPICE_BIN (useful on Windows)."""
    env = os.environ.get("NGSPICE_BIN")
    if env and Path(env).exists():
        return env
    for name in ("ngspice", "ngspice_con", "ngspice_con.exe", "ngspice.exe"):
        p = shutil.which(name)
        if p:
            return p
    raise FileNotFoundError(
        "ngspice not found. Install it (Linux: sudo apt install ngspice | macOS: brew install ngspice | "
        "Windows: https://ngspice.sourceforge.io/download.html) or set NGSPICE_BIN to the executable path."
    )


def _parse(stdout: str) -> dict:
    out = {}
    for line in stdout.splitlines():
        if line.startswith(("OPRES", "ACRES")):
            for k, v in _KV.findall(line):
                out[k] = float(v)
    return out


def run_netlist(netlist: str, timeout: float = 30.0) -> dict:
    """Run a netlist string; returns the raw parsed key/value dict ({} on failure)."""
    exe = find_ngspice()
    with tempfile.TemporaryDirectory(prefix="ota_") as tmp:
        path = Path(tmp) / "ota.sp"
        path.write_text(netlist)
        try:
            res = subprocess.run([exe, "-b", str(path)], capture_output=True, text=True,
                                 timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return {}
    return _parse(res.stdout)


def simulate(design: dict, fault: dict | None = None) -> dict:
    """Simulate one design (optionally with a fault) and return clean measurements.

    Keys: gain_db, ugbw_hz, pm_deg, power_uw, vout, vtail, vd1, isup_ua, ok
    Missing quantities (e.g. no unity-gain crossing in a faulty circuit) are NaN.
    """
    raw = run_netlist(build_netlist(design, fault))
    nan = float("nan")
    res = dict(gain_db=raw.get("dcgain", nan), ugbw_hz=raw.get("ugbw", nan), pm_deg=nan,
               power_uw=nan, vout=raw.get("vout", nan), vtail=raw.get("vtail", nan),
               vd1=raw.get("vd1", nan), isup_ua=nan, ok=int("isup" in raw and "dcgain" in raw))
    if "phugbw" in raw and not math.isnan(res["ugbw_hz"]):
        res["pm_deg"] = 180.0 + raw["phugbw"]
    if "isup" in raw:
        res["isup_ua"] = raw["isup"] * 1e6
        res["power_uw"] = raw["isup"] * design["vdd"] * 1e6
    return res
