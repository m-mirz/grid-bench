"""rustpower: Rust Newton-Raphson (Bevy ECS) with KLU, driven through its Python API.

Input: `rustpower.PowerGrid.from_pandapower` on the pandapower net that
`from_mpc` makes of the prepared `.mat` (the pandapower column's input), so
an importer problem in `from_mpc` shows in both columns. rustpower reads
the net's tables into its own model and builds its own Ybus; it reads bus,
line, trafo, load, gen, ext_grid, shunt, sgen and switch, and nothing else.

Known losses (0.5.2), reported rather than worked around (both traced by
comparing rustpower's `y_bus` with pandapower's on the same net):
- rustpower does not read pandapower's `impedance` table. `from_mpc` writes
  a MATPOWER branch as an `impedance` when it joins buses of different base
  voltage without a tap ratio (case118: 67-115 and 85-86), so those
  branches are missing from rustpower's Ybus. Where that leaves a bus
  isolated, the first factorization fails and `solve` reports no
  convergence after 0 iterations (case118, case300, case1888rte,
  case2848rte, case2869pegase, case6495rte, case9241pegase); where it does
  not, Newton runs to the iteration limit (case18, mvlv1004, mvlv10616,
  mvlv29840, each with 1 to 4 such branches).
- A transformer's series reactance is `sqrt(vk² - vkr²)` (elements/trans.rs),
  which drops the sign of a negative `vk_percent`; pandapower keeps it.
  case3120sp has 10 branches with negative reactance (MATPOWER x < 0, read
  as transformers), so rustpower converges to a different problem and the
  oracle rejects it (923 MW, 10059 Mvar).

Bus ids: rustpower indexes buses by their pandapower index and requires it
to be 0..n-1: `from_mpc` sets it to MATPOWER bus number - 1, which has gaps
on most cases (case300, pegase, RTE, mvlv), and `from_pandapower` then
panics in nalgebra-sparse ("assertion failed: i < self.nrows"). `load`
therefore renumbers the buses with pandapower's `create_continuous_bus_index`,
a relabelling that leaves the power-flow equations untouched, and keeps the
old index in `net.bus.old_index`: the MATPOWER number is `old_index + 1`.
`PowerGrid.v` is in bus-id order (asserted against `n_bus`).

Settings, each the closest match to the common problem definition:
- The wheel from PyPI, built with its `klu` feature (SuiteSparse KLU, as
  `rustpower.features()` reports), not the pure-Rust `rsparse` default of
  the crate: the build the project ships and benchmarks.
- `enable_cache(True)`: keeps the Jacobian sparsity pattern and KLU's
  symbolic factorization across solves of an unchanged topology, as
  lightsim2grid and p3s do; it is how the README times its hot loop, and
  what "warm solves on a persistent model" means here.
- `solve(v_init, MAX_ITERATIONS, TOLERANCE_PU)` with `v_init` all 1.0 p.u.
  and zero angle on every call: a flat start. rustpower re-pins the PV and
  slack magnitudes and the slack angle itself (`set_v` runs `v_inj`), so
  this is the same start as lightsim2grid's. The tolerance is the infinity
  norm of the per-unit power mismatch, as everywhere else.
- Not enabled: `enable_dcpf_init` (a DC start is not a flat start),
  `enable_iwamoto` (step-size control; plain Newton is the common method),
  `qlim` (no reactive limits).
- Divergence is a falsy `SolveReport`, raised here as DidNotConverge.
"""
import numpy as np

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import mat_path


class RustpowerAdapter(SolverAdapter):
    name = "rustpower"
    display_name = "rustpower (KLU)"
    color = "#a94d25"   # tools/palette.py: Rust brown, step 1, dashed
    package = "rustpower"
    modules = ("rustpower", "pandapower", "pandapower.converter.matpower.from_mpc", "pandapower.toolbox")
    language = "rust"
    families = ("matpower", "distribution")
    settings = {"linear_solver": "KLU", "init": "flat", "tolerance_pu": TOLERANCE_PU,
                "max_iteration": MAX_ITERATIONS, "cache": True, "dcpf_init": False, "iwamoto": False,
                "qlim": False, "continuous_bus_index": True}

    def load(self, case):
        import rustpower
        from pandapower.converter.matpower.from_mpc import from_mpc
        from pandapower.toolbox import create_continuous_bus_index
        assert "klu" in rustpower.features(), rustpower.features()
        net = from_mpc(str(mat_path(case)), f_hz=50)
        create_continuous_bus_index(net, store_old_index=True)
        grid = rustpower.PowerGrid.from_pandapower(net)
        grid.enable_cache(True)
        n = len(net.bus)
        assert grid.n_bus == n, (grid.n_bus, n)
        return {"grid": grid, "ids": [str(int(i) + 1) for i in net.bus.old_index],
                "v_init": np.ones(n, dtype=complex), "report": None}

    def solve(self, model):
        report = model["grid"].solve(v_init=model["v_init"], max_iter=MAX_ITERATIONS, tol=TOLERANCE_PU)
        model["report"] = report
        if not report.converged:
            raise DidNotConverge(f"rustpower did not converge after {report.iterations} iterations")

    def solution(self, model, case):
        v = model["grid"].v
        if len(v) != len(model["ids"]):
            raise RuntimeError(f"rustpower has {len(v)} buses, the case {len(model['ids'])}")
        ids = model["ids"]
        return Solution(dict(zip(ids, np.abs(v))), dict(zip(ids, np.angle(v, deg=True))),
                        int(model["report"].iterations))
