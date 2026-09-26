"""pandapower: `pp.runopp`, AC-OPF with PIPS (PYPOWER's primal-dual
interior-point solver, the Python port of MATPOWER's MIPS, in pandapower).

Input: `from_mpc` on the prepared `.mat`, as the power-flow adapter reads it
(bus index = MATPOWER bus number - 1). `from_mpc` imports the generator
costs (`poly_cost`, from `gencost`) and limits. Joins: generators by `.m`
row through `net._from_ppc_lookups["gen"]` (row -> ext_grid or gen index),
the ext_grid's and each gen's bus asserted against the row's.

How pandapower states the case's constraints, reported by the oracle where
it differs, not changed:
- Branch limits: `from_mpc` turns `rateA` into a line's `max_i_ka` (at
  nominal voltage) and a transformer's `sn_mva`, each at
  `max_loading_percent = 100`, and the OPF limits their loading, i.e.
  current: an MVA limit of `rateA` times |V|, not `rateA`.
- Angle-difference limits: not imported, and pandapower's OPF has none, so
  `angmin`/`angmax` are not enforced (the `__sad` cases bind them).

Settings:
- `ext_grid.controllable = True`: `from_mpc` imports the slack as a
  non-controllable ext_grid, which pandapower's OPF holds at its voltage
  setpoint (within `delta`); the case's slack voltage is free within the
  bus's limits and its generator dispatchable, as in MATPOWER. Controllable
  states that problem.
- `init="flat"`: PIPS starts every variable at the middle of its bounds
  (|V| at 1.0 for PGLib's 0.94 to 1.06, generators at the middle of their
  ranges) and every angle at the reference angle: the common flat start.
- `PDIPM_GRADTOL`, `PDIPM_COMPTOL`, `PDIPM_COSTTOL`, `PDIPM_FEASTOL` =
  `OPF_TOLERANCE`, `PDIPM_MAX_IT = OPF_MAX_ITERATIONS`, as MATPOWER's MIPS.
- `calculate_voltage_angles=True`, numba on (as in the power flow).

Result: the optimum on case14 (the same as MATPOWER's to 4e-7). Where a
branch limit binds, pandapower's is a current limit: on case118 both
overloaded branches carry exactly `rateA` times |V| (92.220 and 160.060 MVA
at 1.06 p.u. against 87 and 151), 6 % over the case's MVA limit, at a cost
0.18 % below the reference. The `__sad` cases break their angle limits (by
about 1 degree on case14), 22 % cheaper than the reference. case300 does
not converge from the flat start.
"""
from adapters.optimizer_adapter import OPF_MAX_ITERATIONS, OPF_TOLERANCE, OptimizerAdapter
from adapters.pandapower_adapter import PandapowerAdapter
from adapters.solver_adapter import DidNotConverge, Solution
from cases.matpower import GEN_BUS, GEN_STATUS, parse_m
from cases.registry import CASES, mat_path


class PandapowerOptimizer(OptimizerAdapter):
    name = PandapowerAdapter.name
    display_name = PandapowerAdapter.display_name
    color = PandapowerAdapter.color
    package = PandapowerAdapter.package
    language = PandapowerAdapter.language
    modules = ("pandapower", "pandapower.converter.matpower.from_mpc")
    settings = {"solver": "PIPS", "init": "flat", "ext_grid_controllable": True, "branch_limit": "current (loading %)",
                "angle_limits": False, "tolerance": OPF_TOLERANCE, "max_it": OPF_MAX_ITERATIONS, "numba": True}

    def load(self, case):
        from pandapower.converter.matpower.from_mpc import from_mpc
        net = from_mpc(str(mat_path(case)), f_hz=50)
        net.ext_grid["controllable"] = True
        gen = parse_m(CASES[case]["file"])["gen"]
        lookup = net._from_ppc_lookups["gen"]  # noqa: SLF001, the importer's own row -> element map
        rows = []
        for row in range(len(gen)):
            if gen[row, GEN_STATUS] <= 0:
                continue
            et, idx = lookup.at[row, "element_type"], int(lookup.at[row, "element"])
            assert int(net[et].at[idx, "bus"]) == int(gen[row, GEN_BUS]) - 1
            rows.append((str(row), et, idx))
        return {"net": net, "gens": rows}

    def solve(self, model):
        import pandapower as pp
        try:
            pp.runopp(model["net"], init="flat", calculate_voltage_angles=True, numba=True,
                      PDIPM_GRADTOL=OPF_TOLERANCE, PDIPM_COMPTOL=OPF_TOLERANCE, PDIPM_COSTTOL=OPF_TOLERANCE,
                      PDIPM_FEASTOL=OPF_TOLERANCE, PDIPM_MAX_IT=OPF_MAX_ITERATIONS)
        except pp.OPFNotConverged as e:
            raise DidNotConverge(str(e) or "runopp: OPF did not converge") from e

    def solution(self, model, case):
        net = model["net"]
        res = net.res_bus
        ids = [str(int(i) + 1) for i in res.index]
        pg = {row: float(net[f"res_{et}"].at[idx, "p_mw"]) for row, et, idx in model["gens"]}
        qg = {row: float(net[f"res_{et}"].at[idx, "q_mvar"]) for row, et, idx in model["gens"]}
        return Solution(dict(zip(ids, res.vm_pu)), dict(zip(ids, res.va_degree)), pg_mw=pg, qg_mvar=qg)
