"""Fill the ngspice netlist template from a design dictionary (+ optional fault)."""
from __future__ import annotations

from string import Template

from . import config as C

_TEMPLATE = None


def _template() -> Template:
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = Template(C.NETLIST_TEMPLATE.read_text())
    return _TEMPLATE


def build_netlist(design: dict, fault: dict | None = None) -> str:
    f = fault or dict(ibias_scale=1.0, dvt_in=0.0, cl_scale=1.0, m4gate="d1", fault_lines="")
    d = design
    vtn = C.VTN0 + d.get("dvtn", 0.0)
    vtp = C.VTP0 + d.get("dvtp", 0.0)
    kpn = C.KPN0 * (1 + d.get("dkpn", 0.0))
    kpp = C.KPP0 * (1 + d.get("dkpp", 0.0))
    params = dict(
        temp=d["temp"], vdd=d["vdd"],
        vtn=f"{vtn:.5g}", vtn_in=f"{vtn + f['dvt_in']:.5g}", vtp=f"{vtp:.5g}",
        kpn=f"{kpn:.5g}", kpp=f"{kpp:.5g}",
        lam_in=f"{C.LAMBDA_N / d['l1_um']:.5g}",
        lam_ref=f"{C.LAMBDA_N / C.L_REF_UM:.5g}",
        lam_ld=f"{C.LAMBDA_P / d['l3_um']:.5g}",
        ibias=f"{d['ibias_ua'] * 1e-6 * f['ibias_scale']:.6g}",
        w0=f"{C.W_REF_UM:.5g}u", w5=f"{C.W_REF_UM * d['ratio']:.5g}u", l0=f"{C.L_REF_UM:.5g}u",
        vcm=f"{C.VCM_FRAC * d['vdd']:.5g}",
        w1=f"{d['w1_um']:.5g}u", l1=f"{d['l1_um']:.5g}u",
        w3=f"{d['w3_um']:.5g}u", l3=f"{d['l3_um']:.5g}u",
        m4gate=f["m4gate"], cl=f"{d['cl_pf'] * f['cl_scale']:.5g}p",
        fault_lines=f["fault_lines"],
    )
    # safe_substitute: the .control block contains "$&vector" tokens that must stay untouched
    return _template().safe_substitute(params)
