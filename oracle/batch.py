"""Grading a batch power flow: tier 1 (`oracle.residual`) on every scenario.

A batch solve returns one voltage set per scenario. Each is checked against
the case that scenario defines, built from the raw `.m`: an operating point
of a sweep (`cases.sweep.scenario`) or an N-1 outage
(`cases.contingency.outage`). A tool that applies a scenario wrongly (a load
left at its base value, the wrong branch taken out) shows a residual on
exactly those scenarios. Grading only a sample would let a race between
threads, or a batch API that silently reuses one scenario's result, pass.

The record says how many scenarios failed and which was worst; the residual
fields are the maxima over all scenarios (`residual_worst_bus` and
`residual_failing` those of the worst scenario), so `oracle_ok` holds only
if every scenario meets the single-solve tolerances.
"""
from collections.abc import Callable

import numpy as np

from oracle.residual import residual


def check(scenario: Callable[[int], dict], n: int, bus_ids: list[str], vm: np.ndarray, va_deg: np.ndarray,
          tol_mva: float, tol_pu: float) -> dict:
    """`scenario(k)`: the case of scenario k, for k in range(n). `vm`,
    `va_deg`: scenarios x `bus_ids` (MATPOWER bus numbers as str)."""
    assert vm.shape == va_deg.shape == (n, len(bus_ids)), f"expected {n} scenarios of {len(bus_ids)} buses"
    worst = {"max_dp_mw": 0.0, "max_dq_mvar": 0.0, "max_dvm_pu": 0.0}
    failed, worst_k, worst_mva, worst_r, n_checked = 0, 0, -1.0, None, None
    for k in range(n):
        r = residual(scenario(k), dict(zip(bus_ids, vm[k])), dict(zip(bus_ids, va_deg[k])),
                     tol_mva=tol_mva, tol_pu=tol_pu)
        for f in worst:
            worst[f] = max(worst[f], getattr(r, f))
        mva = max(r.max_dp_mw, r.max_dq_mvar)
        if worst_r is None or mva > worst_mva:
            worst_k, worst_mva, worst_r = k, mva, r
        ok = mva <= tol_mva and r.max_dvm_pu <= tol_pu and r.n_checked == r.n_buses
        failed += not ok
        n_checked = r.n_checked if n_checked is None else min(n_checked, r.n_checked)
    return {**{f"residual_{f}": v for f, v in worst.items()},
            "residual_n_checked": n_checked, "residual_n_buses": worst_r.n_buses,
            "residual_worst_bus": worst_r.worst_bus, "residual_failing": worst_r.failing,
            "scenarios": n, "scenarios_failed": failed, "worst_scenario": worst_k, "oracle_ok": failed == 0}
