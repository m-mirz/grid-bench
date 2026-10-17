"""Grading a batch power flow: tier 1 (`oracle.residual`) on every scenario.

A batch solve returns one voltage set per scenario. Each is checked against
the case that scenario defines, built from the raw `.m`: an operating point
of a sweep (`cases.sweep.scenario`) or an N-1 outage
(`cases.contingency.outage`). A tool that applies a scenario wrongly (a load
left at its base value, the wrong branch taken out) shows a residual on
exactly those scenarios. Grading only a sample would let a race between
threads, or a batch API that silently reuses one scenario's result, pass.

The record says how many scenarios failed and which was worst; the residual
fields are the maxima over all scenarios (`residual_min_vm_pu` the minimum,
`residual_worst_bus` and `residual_failing` those of the worst scenario), so `oracle_ok` holds only
if every scenario meets the single-solve tolerances.

With `workers` > 1 the scenarios' residuals are computed in that many
processes (`spawn`: the caller may hold a Julia runtime, a CUDA context or a
JVM, which a fork would copy), then reduced here in scenario order, so the
record is the one a serial check writes. `scenario` must then be picklable
(a module-level function or a `functools.partial` of one, not a lambda).
"""
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from oracle.residual import Residual, residual


def check(scenario: Callable[[int], dict], n: int, bus_ids: list[str], vm: np.ndarray, va_deg: np.ndarray,
          tol_mva: float, tol_pu: float, vm_floor: float, workers: int = 1) -> dict:
    """`scenario(k)`: the case of scenario k, for k in range(n). `vm`,
    `va_deg`: scenarios x `bus_ids` (MATPOWER bus numbers as str)."""
    assert vm.shape == va_deg.shape == (n, len(bus_ids)), f"expected {n} scenarios of {len(bus_ids)} buses"
    if workers > 1:
        # About eight chunks per worker: even load, and each chunk's voltages pickled once.
        bounds = np.linspace(0, n, min(n, 8 * workers) + 1).astype(int)
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            parts = pool.map(_residuals, [(scenario, a, b, bus_ids, vm[a:b], va_deg[a:b], tol_mva, tol_pu)
                                          for a, b in zip(bounds[:-1], bounds[1:])])
            results = [r for part in parts for r in part]
    else:
        results = _residuals((scenario, 0, n, bus_ids, vm, va_deg, tol_mva, tol_pu))
    worst = {"max_dp_mw": 0.0, "max_dq_mvar": 0.0, "max_dvm_pu": 0.0}
    failed, worst_k, worst_mva, worst_r, n_checked, min_vm = 0, 0, -1.0, None, None, np.inf
    for k, r in enumerate(results):
        for f in worst:
            worst[f] = max(worst[f], getattr(r, f))
        mva = max(r.max_dp_mw, r.max_dq_mvar)
        if worst_r is None or mva > worst_mva:
            worst_k, worst_mva, worst_r = k, mva, r
        min_vm = min(min_vm, r.min_vm_pu)
        ok = mva <= tol_mva and r.max_dvm_pu <= tol_pu and r.n_checked == r.n_buses and r.min_vm_pu >= vm_floor
        failed += not ok
        n_checked = r.n_checked if n_checked is None else min(n_checked, r.n_checked)
    return {**{f"residual_{f}": v for f, v in worst.items()},
            "residual_min_vm_pu": min_vm, "residual_n_checked": n_checked, "residual_n_buses": worst_r.n_buses,
            "residual_worst_bus": worst_r.worst_bus, "residual_failing": worst_r.failing,
            "scenarios": n, "scenarios_failed": failed, "worst_scenario": worst_k, "oracle_ok": failed == 0}


def _residuals(args) -> list[Residual]:
    """Scenarios a..b-1, their voltages the rows of `vm`, `va_deg`."""
    scenario, a, b, bus_ids, vm, va_deg, tol_mva, tol_pu = args
    return [residual(scenario(k), dict(zip(bus_ids, vm[i])), dict(zip(bus_ids, va_deg[i])),
                     tol_mva=tol_mva, tol_pu=tol_pu) for i, k in enumerate(range(a, b))]
