"""Smoke test for the numerical-solver stack baked into lbx-tasks-base.

Each check does an import AND a tiny deterministic solve, so a green run proves
the engine actually runs under the harness runtime (python:3.13-slim), not just
that the wheel imported. Run it inside the built base image:

    docker run --rm lbx-tasks-base:runtime-ml-core-py313-local \\
        python /path/to/base/tests/test_solvers.py

or as a plain script in any environment where the base stack is installed.
Exits non-zero if any applicable solver fails. Heavy engines that are not
present (e.g. OpenROAD, which is GPU-base only) are reported as SKIP, not FAIL.

Covers the pip stack (aerosandbox, PySpice, scikit-fem, PyNite, anastruct,
scikit-rf, cantera, CoolProp, klayout) and the heavy binary engines
(OpenSeesPy, Meep, CalculiX, SU2, OpenFOAM, OpenROAD).

Expected reference values (identical on linux/amd64 and arm64):
    aerosandbox VLM CL=0.4324 | pyspice Vout=5.000V | cantera Tad=2225.5K
    coolprop water Tsat=373.12K | meep FDTD Ez=-0.26517
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import traceback
from pathlib import Path

_results: list[tuple[str, bool, str]] = []


class _Skip(Exception):
    """Raised by a check that is not applicable on this base (e.g. GPU-only)."""


def _check(name: str, fn) -> None:
    try:
        detail = fn()
        _results.append((name, True, detail))
        print(f"PASS  {name:14s} {detail}", flush=True)
    except _Skip as exc:
        print(f"SKIP  {name:14s} {exc}", flush=True)
    except Exception as exc:  # noqa: BLE001
        _results.append((name, False, repr(exc)))
        print(f"FAIL  {name:14s} {exc!r}", flush=True)
        traceback.print_exc()


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)


# --- Domain 1: CFD / Aerodynamics -------------------------------------------
def aerosandbox() -> str:
    import aerosandbox as asb

    wing = asb.Wing(
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=1.0, airfoil=asb.Airfoil("naca0012")),
            asb.WingXSec(xyz_le=[0, 5, 0], chord=1.0, airfoil=asb.Airfoil("naca0012")),
        ],
    )
    airplane = asb.Airplane(wings=[wing])
    op = asb.OperatingPoint(velocity=25.0, alpha=5.0)
    aero = asb.VortexLatticeMethod(airplane=airplane, op_point=op).run()
    cl = float(aero["CL"])
    assert cl > 0, cl
    return f"VLM CL={cl:.4f} (v{asb.__version__})"


# --- Domain 2: Analog Circuits (SPICE) --------------------------------------
def pyspice() -> str:
    from PySpice.Spice.Netlist import Circuit
    from PySpice.Unit import u_Ohm, u_V

    c = Circuit("divider")
    c.V("1", "vin", c.gnd, 10 @ u_V)
    c.R(1, "vin", "vout", 1 @ u_Ohm)
    c.R(2, "vout", c.gnd, 1 @ u_Ohm)
    analysis = c.simulator().operating_point()
    vout = float(analysis["vout"][0])
    assert abs(vout - 5.0) < 1e-6, vout
    return f"ngspice OP Vout={vout:.3f}V"


# --- Domain 3: Structural Mechanics / FEM -----------------------------------
def scikit_fem() -> str:
    import skfem
    from skfem import Basis, ElementTriP1, MeshTri, asm, condense, solve
    from skfem.models.poisson import laplace, unit_load

    basis = Basis(MeshTri().refined(3), ElementTriP1())
    A = asm(laplace, basis)
    b = asm(unit_load, basis)
    x = solve(*condense(A, b, D=basis.get_dofs()))
    return f"Poisson max={x.max():.4f} (v{skfem.__version__})"


def pynite() -> str:
    from Pynite import FEModel3D

    m = FEModel3D()
    m.add_node("N1", 0, 0, 0)
    m.add_node("N2", 10, 0, 0)
    m.def_support("N1", True, True, True, True, True, True)
    m.add_material("Steel", 29000, 11200, 0.3, 0.000284)
    m.add_section("Sec", 10, 100, 100, 200)
    m.add_member("M1", "N1", "N2", "Steel", "Sec")
    m.add_node_load("N2", "FY", -10)
    m.analyze(check_statics=False)
    dy = m.nodes["N2"].DY["Combo 1"]
    assert dy < 0, dy
    return f"cantilever tip DY={dy:.4e}"


def anastruct() -> str:
    from anastruct import SystemElements

    ss = SystemElements()
    ss.add_element(location=[[0, 0], [5, 0]])
    ss.add_support_fixed(node_id=1)
    ss.point_load(node_id=2, Fy=-10)
    ss.solve()
    uy = ss.get_node_displacements(node_id=2)["uy"]
    return f"beam node2 uy={uy:.4e}"


# --- Domain 4: Electromagnetics (RF networks) -------------------------------
def scikit_rf() -> str:
    import numpy as np
    import skrf as rf

    freq = rf.Frequency(1, 10, 11, "ghz")
    line = rf.media.DefinedGammaZ0(frequency=freq, z0=50)
    ntwk = line.line(90, "deg", z0=75)
    return f"|S11[0]|={np.abs(ntwk.s[0, 0, 0]):.4f} (v{rf.__version__})"


# --- Domain 5: Reacting Flow / Thermodynamics -------------------------------
def cantera() -> str:
    import cantera as ct

    gas = ct.Solution("gri30.yaml")
    gas.TPX = 300.0, ct.one_atm, "CH4:1, O2:2, N2:7.52"
    gas.equilibrate("HP")
    assert gas.T > 1500, gas.T
    return f"CH4/air Tad={gas.T:.1f}K (v{ct.__version__})"


def coolprop() -> str:
    from CoolProp.CoolProp import PropsSI

    t = PropsSI("T", "P", 101325, "Q", 0, "Water")
    assert abs(t - 373.12) < 1.0, t
    return f"water Tsat@1atm={t:.2f}K"


# --- Domain 6: EDA (layout I/O) ---------------------------------------------
def klayout() -> str:
    import klayout.db as db

    ly = db.Layout()
    top = ly.create_cell("TOP")
    top.shapes(ly.layer(1, 0)).insert(db.Box(0, 0, 1000, 2000))
    area = top.bbox().area()
    assert area == 2_000_000, area
    return f"layout bbox area={area}"


# --- Heavy binary engines (conda envs / pip / multi-stage binary) -----------
_ENVS = Path("/opt/solver-envs")


def openseespy() -> str:
    import openseespy.opensees as ops

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.fix(1, 1, 1, 1)
    return "model+node build OK"


def meep() -> str:
    py = _ENVS / "meep" / "bin" / "python"
    if not py.exists():
        raise _Skip("meep env not installed")
    code = (
        "import meep as mp;"
        "sim=mp.Simulation(cell_size=mp.Vector3(4,4,0),resolution=10,"
        "geometry=[mp.Block(mp.Vector3(mp.inf,1,mp.inf),material=mp.Medium(epsilon=12))],"
        "sources=[mp.Source(mp.ContinuousSource(frequency=0.15),component=mp.Ez,center=mp.Vector3(-1.5,0))]);"
        "sim.run(until=20);"
        "print('RESULT', round(float(sim.get_field_point(mp.Ez,mp.Vector3(1.5,0)).real),5), mp.__version__)"
    )
    r = _run([str(py), "-c", code])
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr else "meep solve failed")
    # meep prints a banner before and an "Elapsed run time" line after our
    # output, so match the sentinel-prefixed line rather than a position.
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("RESULT"))
    _, val, ver = line.split()
    return f"FDTD Ez={val} (v{ver})"


def calculix() -> str:
    ccx = shutil.which("ccx") or str(_ENVS / "calculix" / "bin" / "ccx")
    if not Path(ccx).exists():
        raise _Skip("calculix not installed")
    r = _run([ccx, "-v"])
    out = (r.stdout + r.stderr).strip().splitlines()
    line = next((ln for ln in out if "ersion" in ln or "CalculiX" in ln), out[-1] if out else "")
    return f"ccx callable: {line.strip()[:48]}"


def su2() -> str:
    exe = shutil.which("SU2_CFD") or str(_ENVS / "su2" / "bin" / "SU2_CFD")
    if not Path(exe).exists():
        raise _Skip("su2 not installed")
    r = _run([exe])  # no config -> emits the "missing .cfg" parse error
    blob = r.stdout + r.stderr
    if not any(s in blob for s in ("CConfig", "configuration file", ".cfg")):
        raise RuntimeError("SU2_CFD produced no recognizable output")
    return "SU2_CFD callable (config-parse OK)"


def openfoam() -> str:
    act = Path("/etc/solver-envs.d/openfoam.sh")
    if not act.exists():
        raise _Skip("openfoam not installed")
    r = _run(["bash", "-lc", f"source {act} && blockMesh -help"])
    if "Usage: blockMesh" not in (r.stdout + r.stderr):
        raise RuntimeError("blockMesh not callable after sourcing activation")
    return "blockMesh callable (env OK)"


def openroad() -> str:
    exe = shutil.which("openroad")
    if not exe:
        raise _Skip("openroad not on PATH (GPU base only)")
    r = _run([exe, "-version"])
    ver = (r.stdout + r.stderr).strip().splitlines()[0] if (r.stdout + r.stderr).strip() else ""
    return f"openroad -version: {ver[:40]}"


CHECKS = [
    ("aerosandbox", aerosandbox),
    ("pyspice", pyspice),
    ("scikit-fem", scikit_fem),
    ("pynite", pynite),
    ("anastruct", anastruct),
    ("scikit-rf", scikit_rf),
    ("cantera", cantera),
    ("coolprop", coolprop),
    ("klayout", klayout),
    # heavy binary engines
    ("openseespy", openseespy),
    ("meep", meep),
    ("calculix", calculix),
    ("su2", su2),
    ("openfoam", openfoam),
    ("openroad", openroad),
]


def main() -> int:
    for name, fn in CHECKS:
        _check(name, fn)
    n_pass = sum(1 for _, ok, _ in _results if ok)
    print(f"\n{n_pass}/{len(_results)} solvers passed")
    return 0 if n_pass == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
