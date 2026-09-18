"""One entry point: grade a tool's solution against its case. Tool-free.

Returns flat fields that go straight into a result record's `extra_info`:

- matpower: tier 1, the power-flow residual against the `.m` (see
  `oracle.residual`). `oracle_ok` holds when every checked bus meets
  `RESIDUAL_OK_MVA` and `SETPOINT_OK_PU`. For cases with phase shifters the
  residual against the zero-shift network is recorded too: a tool that
  cannot represent phase shift (power-grid-model via its converter) passes
  that one and fails the real one, which identifies the loss.
- cgmes: tier 2, deviation from the published SvVoltage (`oracle.cgmes_sv`).
- converted-*: MATPOWER cases converted to CGMES. The tool's TopologicalNode
  voltages are mapped back to MATPOWER buses (`oracle.cgmes_model`) and graded
  by tier 1 against the original `.m`: the converter's losses and the
  tool's importer losses both show up here, and `oracle.cgmes_model.fidelity`
  separates the converter's share.
"""
from functools import lru_cache

from cases.matpower import ANGLE, parse_m
from cases.registry import CASES, cgmes_files, cgmes_sv_file
from oracle import cgmes_model, cgmes_sv
from oracle.residual import residual

RESIDUAL_OK_MVA = 1e-3   # all tools are asked to converge to 1e-8 p.u. (1e-6 MVA on 100 MVA)
SETPOINT_OK_PU = 1e-6


@lru_cache(maxsize=None)
def _mpc(case: str) -> dict:
    return parse_m(CASES[case]["file"])


@lru_cache(maxsize=None)
def _sv(case: str) -> dict:
    return cgmes_sv.published_voltages(cgmes_sv_file(case))


@lru_cache(maxsize=None)
def _cgmes_objects(case: str) -> dict:
    return cgmes_model.read(cgmes_files(case))


def evaluate(case: str, vm: dict[str, float], va_deg: dict[str, float]) -> dict:
    if CASES[case]["family"] == "cgmes":
        return cgmes_sv.deviation(_sv(case), vm, va_deg)
    if "source_case" in CASES[case]:
        vm, va_deg = cgmes_model.to_matpower_ids(_cgmes_objects(case), vm, va_deg)
        case = CASES[case]["source_case"]
    mpc = _mpc(case)
    r = residual(mpc, vm, va_deg, tol_mva=RESIDUAL_OK_MVA, tol_pu=SETPOINT_OK_PU)
    out = {f"residual_{k}": v for k, v in r.asdict().items()}
    out["oracle_ok"] = bool(max(r.max_dp_mw, r.max_dq_mvar) <= RESIDUAL_OK_MVA and r.max_dvm_pu <= SETPOINT_OK_PU
                            and r.n_checked == r.n_buses)
    if (mpc["branch"][:, ANGLE] != 0).any():
        z = residual(mpc, vm, va_deg, zero_phase_shifts=True)
        out["residual_zero_shift_max_dp_mw"] = z.max_dp_mw
        out["residual_zero_shift_max_dq_mvar"] = z.max_dq_mvar
    return out
