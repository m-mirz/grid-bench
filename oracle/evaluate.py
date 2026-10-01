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
- se-*: state estimation, graded against the weighted least-squares problem
  of the case's measurement set (`oracle.wls`).
- opf-*: AC optimal power flow, graded for feasibility against the `.m` and
  for cost against PGLib's reference (`oracle.opf`).
- sweep-*, n1-*: batch power flow and N-1 contingencies, tier 1 on every
  scenario (`oracle.batch`), through `evaluate_batch`: the solution is one
  voltage set per scenario.
"""
from functools import lru_cache

import numpy as np

from cases import contingency, sweep
from cases.matpower import ANGLE, parse_m
from cases.pglib import reference_objective
from cases.registry import CASES, cgmes_files, cgmes_sv_file, contingency_path, measurements_path, sweep_path
from oracle import batch, cgmes_model, cgmes_sv, opf, wls
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


def evaluate(case: str, vm: dict[str, float], va_deg: dict[str, float],
             pg_mw: dict[str, float] | None = None, qg_mvar: dict[str, float] | None = None) -> dict:
    if CASES[case]["problem"] == "opf":
        return opf.check(_mpc(case), reference_objective(case), vm, va_deg, pg_mw or {}, qg_mvar or {})
    if CASES[case]["problem"] == "se":
        return wls.check(CASES[case]["file"], measurements_path(case), vm, va_deg)
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


def evaluate_batch(case: str, bus_ids: list[str], vm: np.ndarray, va_deg: np.ndarray) -> dict:
    """A sweep's or a contingency case's solution: scenarios x `bus_ids`,
    graded scenario by scenario."""
    mpc = _mpc(case)
    if CASES[case]["problem"] == "n1":
        data = contingency.read(contingency_path(case))
        scenario, n = (lambda k: contingency.outage(mpc, data, k)), len(data["branch_row"])
    else:
        data = sweep.read(sweep_path(case))
        scenario, n = (lambda k: sweep.scenario(mpc, data, k)), len(data["scale"])
    return batch.check(scenario, n, bus_ids, vm, va_deg, tol_mva=RESIDUAL_OK_MVA, tol_pu=SETPOINT_OK_PU)
