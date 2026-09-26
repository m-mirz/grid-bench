"""pandapower: `pandapower.estimation.estimate`, WLS.

Input: the network as the power-flow adapter reads it (`from_mpc` on the
prepared `.mat`, bus index = MATPOWER bus number - 1), plus the case's
measurement set written into `net.measurement` in one table (the same rows
`create_measurement` appends one at a time; one at a time would time
pandas, not pandapower, on the 30k-bus grids). Joins:
- bus measurements: `element` = MATPOWER bus number - 1;
- branch measurements: the MATPOWER branch row through
  `net._from_ppc_lookups["branch"]` (row -> line or trafo index), asserted
  against the measurement's from and to bus. `side` is the end whose bus is
  the MATPOWER from bus: "from"/"to" on a line, "hv"/"lv" on a transformer
  (`from_mpc` puts hv on the higher-kV end, whichever end that is).
  `create_measurement` also documents the bus index as a valid `side`, but
  the 3.3.3 estimator only maps the strings and silently drops such a
  measurement (found here: with flows given by bus index, the noisy cases
  missed the WLS optimum and exact cases still passed, since the injections
  alone determine the state). `from_mpc` turns some branches into `impedance`
  elements, which pandapower's estimator cannot take a measurement on: such
  a case raises CaseUnsupported.
- Bus P and Q are in load reference ("consistent to the rest of
  pandapower", `create_measurement`), so the injections are negated.

Settings:
- `algorithm="wls"`: plain weighted least squares.
- `init="flat"`: every solve starts at 1 p.u., 0 degrees.
- `zero_injection=None`: the default `"aux_bus"` only affects auxiliary
  buses, which these networks have none of, but None states it: no
  zero-injection constraints or virtual measurements.
- `tolerance=SE_TOLERANCE` (on the maximum state change),
  `maximum_iterations=MAX_ITERATIONS`.
- Not used: `remove_bad_data` / `chi2_analysis`, bad-data handling.
The angle reference is the ext_grid bus, the MATPOWER slack.

numpy: pandapower 3.3.3's estimator calls `np.in1d` and
`np.linalg.linalg`, both removed in numpy 2.4, and declares no upper bound;
the image pins numpy < 2.4 (`tool-configs/pandapower/pyproject.toml`).

Result: the WLS optimum on every case its importer keeps intact (J = 41.4 on
case14~noisy, the oracle tests' own Gauss-Newton value); it misses it on
case300 and case3120sp, where `from_mpc` changes the problem (see the
power-flow adapter). It does not converge from flat start in 30 iterations
on case2848rte, case3120sp and case9241pegase (`~exact`). On mvlv29840 it
asks for a 59.7 GiB dense array and fails: after a successful estimate it
keeps the gain and Jacobian matrices as dense arrays (`toarray()` in
`WLSAlgorithm.estimate`, for its chi-square tests).
"""
import numpy as np

from adapters.estimator_adapter import SE_TOLERANCE, EstimatorAdapter
from adapters.pandapower_adapter import PandapowerAdapter
from adapters.solver_adapter import MAX_ITERATIONS, CaseUnsupported, DidNotConverge, Solution
from cases import measurements
from cases.registry import CASES, mat_path, measurements_path

_TYPE = {"vm": "v", "p_inj": "p", "q_inj": "q", "p_from": "p", "q_from": "q"}


class PandapowerEstimator(EstimatorAdapter):
    name = PandapowerAdapter.name
    display_name = PandapowerAdapter.display_name
    color = PandapowerAdapter.color
    package = PandapowerAdapter.package
    language = PandapowerAdapter.language
    modules = ("pandapower", "pandapower.converter.matpower.from_mpc", "pandapower.estimation", "pandas")
    settings = {"algorithm": "wls", "init": "flat", "zero_injection": None,
                "tolerance": SE_TOLERANCE, "maximum_iterations": MAX_ITERATIONS}

    def load(self, case):
        import pandas as pd
        from pandapower.converter.matpower.from_mpc import from_mpc
        net = from_mpc(str(mat_path(CASES[case]["base_case"])), f_hz=50)
        lookup = net._from_ppc_lookups["branch"]  # noqa: SLF001, the importer's own row -> element map
        rows = []
        for m in measurements.read(measurements_path(case))["measurements"]:
            kind = _TYPE[m["kind"]]
            if "bus" in m:
                value = -m["value"] if kind in ("p", "q") else m["value"]
                rows.append((kind, "bus", m["bus"] - 1, value, m["sigma"], None))
                continue
            et, idx = lookup.at[m["branch_row"], "element_type"], int(lookup.at[m["branch_row"], "element"])
            if et not in ("line", "trafo"):
                raise CaseUnsupported(f"branch row {m['branch_row']} became a pandapower {et}, "
                                      "which the estimator takes no measurement on")
            ends = {"from": "from_bus", "to": "to_bus"} if et == "line" else {"hv": "hv_bus", "lv": "lv_bus"}
            at = {int(net[et].at[idx, col]): side for side, col in ends.items()}
            assert set(at) == {m["from_bus"] - 1, m["to_bus"] - 1}
            rows.append((kind, et, idx, m["value"], m["sigma"], at[m["from_bus"] - 1]))
        table = pd.DataFrame(rows, columns=["measurement_type", "element_type", "element", "value", "std_dev", "side"])
        table.insert(0, "name", None)
        net.measurement = table.astype(net.measurement.dtypes.to_dict() | {"side": object})
        return net

    def solve(self, net):
        from pandapower.estimation import estimate
        # 3.3.3 returns a dict (always truthy), not the bool its docstring names
        out = estimate(net, algorithm="wls", init="flat", tolerance=SE_TOLERANCE,
                       maximum_iterations=MAX_ITERATIONS, zero_injection=None)
        if not out["success"]:
            raise DidNotConverge(f"WLS not converged in {out['num_iterations']} iterations")
        self._iterations = int(out["num_iterations"])

    def solution(self, net, case):
        res = net.res_bus_est
        ids = [str(int(i) + 1) for i in res.index]
        ok = np.isfinite(res.vm_pu.values)
        return Solution({b: v for b, v, k in zip(ids, res.vm_pu, ok) if k},
                        {b: a for b, a, k in zip(ids, res.va_degree, ok) if k}, self._iterations)
