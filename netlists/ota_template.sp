* 5T OTA (NMOS input pair, PMOS mirror load, NMOS mirror-biased tail)
* Placeholders (dollar-prefixed words) are filled in by otaml/netlist.py
.title 5T_OTA
.temp $temp
.options gmin=1e-12 reltol=1e-4 abstol=1e-13 vntol=1e-7

* ---- Level-1 models (self-contained, no external PDK needed) ----
.model nm_in  nmos level=1 vto=$vtn_in kp=$kpn gamma=0.4 phi=0.7 lambda=$lam_in  tox=4e-9 cgso=3e-10 cgdo=3e-10 cj=5e-4 tnom=27
.model nm_ref nmos level=1 vto=$vtn kp=$kpn gamma=0.4 phi=0.7 lambda=$lam_ref tox=4e-9 cgso=3e-10 cgdo=3e-10 cj=5e-4 tnom=27
.model pm_ld  pmos level=1 vto=-$vtp kp=$kpp gamma=0.4 phi=0.7 lambda=$lam_ld tox=4e-9 cgso=3e-10 cgdo=3e-10 cj=5e-4 tnom=27

* ---- supplies and bias ----
VDD  vdd 0 DC $vdd
IREF vdd nb DC $ibias
M0   nb  nb 0 0 nm_ref W=$w0 L=$l0
M5   tail nb 0 0 nm_ref W=$w5 L=$l0

* ---- inputs (AC stimulus on the non-inverting input) ----
VINN inn 0 DC $vcm
VINP inp 0 DC $vcm AC 1

* ---- core ----
M1 d1  inp tail 0 nm_in W=$w1 L=$l1
M2 out inn tail 0 nm_in W=$w1 L=$l1
M3 d1  d1  vdd vdd pm_ld W=$w3 L=$l3
M4 out $m4gate vdd vdd pm_ld W=$w3 L=$l3
CL out 0 $cl
$fault_lines

.control
set noaskquit
set units=degrees
op
let isup = -i(vdd)
let vo = v(out)
let vt = v(tail)
let vd = v(d1)
echo OPRES isup=$&isup vout=$&vo vtail=$&vt vd1=$&vd
ac dec 20 10 10G
meas ac dcgain find vdb(out) at=10
meas ac ugbw when vdb(out)=0 fall=1
let ph = cph(out)
meas ac phugbw find ph at=ugbw
echo ACRES dcgain=$&dcgain ugbw=$&ugbw phugbw=$&phugbw
.endc
.end
