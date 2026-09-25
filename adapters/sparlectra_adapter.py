"""Sparlectra: Sparlectra.jl's rectangular complex-state Newton-Raphson
(Julia, UMFPACK sparse LU), driven from Python through juliacall, which runs
Julia inside the benchmark process: a call costs microseconds, so the timing
is Julia's.

Version: 0.17.3, the newest release when it was added (2026-09-25), chosen
deliberately over the 7-day rule every other pin follows (the package
releases almost daily); tool-configs/sparlectra/julia/setup.jl says so too.

Inputs:
- matpower, distribution: `createNetFromMatPowerFile` on the original `.m`
  (Sparlectra's own MATPOWER parser; it maps `baseKV == 0` itself). Every
  import option that has a default in Sparlectra's configuration file is
  passed explicitly, so no configuration file can change the problem:
  `matpower_shift_sign=1`, `matpower_shift_unit=:deg`, `matpower_ratio=:normal`
  (MATPOWER's own conventions), `tap_changer_model=:ideal` (the tap only
  changes the ratio; the case's r and x are used as given),
  `matpower_pv_voltage_source=:gen_vg` (the generator's Vg is the PV
  setpoint, as MATPOWER uses it), `bus_shunt_model=:admittance`,
  `matpower_dcline_mode=:pf_injections` (the default). None of Sparlectra's
  per-case sidecar conventions: every tool solves the case as written.
  Bus mapping: `Net.busOrigIdxDict`, the MATPOWER bus number `addBus!`
  records for each node, asserted complete and unique.
- cgmes, converted-*: `importCGMES` on the zip of the case's profiles built
  by `cases.prep` (never the SV). Bus mapping: the importer names each bus
  after its TopologicalNode (`result.topo.bus_name`); `net.busDict` gives the
  node. Per-side boundary buses and auxiliary buses have no TN and are left
  out (the oracle's `n_checked` shows it).

Import settings (CGMES):
- `tap_control=false`: no tap-changer outer loop; the SSH tap positions are
  used (the SvTapStep positions would be, but no SV is given).
- `machine_control=false`: see the remote-regulation finding below.
- `multi_slack=false`: a single slack, chosen by the importer from
  `referencePriority`, else the largest machine.
- `require_boundary=true`, `hvdc_mode=:injections` (defaults).

Solve settings (`runpf_rectangular!`):
- `damp=1.0`: a full Newton step. The keyword form defaults to 0.2, a damped
  iteration that needs several times the steps.
- `autodamp`, `merit_enabled`, `trust_region_enabled` off: no step control,
  plain Newton-Raphson, as the other tools do.
- Flat start on every solve, as MATPOWER's runpf defines it: PQ buses at
  1 p.u., PV and slack buses at their generators' setpoints, every angle 0.
  `opt_flatstart=true` alone is not that: it takes PV magnitudes and the
  slack angle from the node, i.e. from the case's bus VM/VA columns on
  import and from the previous solution afterwards, so the adapter writes the
  start into the nodes before every solve.
- The imported bus types are restored before every solve too: a solve
  writes its internal bus types back to the `Net`, and it carries isolated
  buses internally as PQ, so it turns every Isolated bus into PQ. The next
  solve on the same `Net` then includes buses without a branch and diverges
  (cgmes_smallgrid: 5 steps on the first solve, divergence on every later
  one). Restoring undoes the side effect; the problem is unchanged.
- `start_projection`, `start_current_iteration_enabled`,
  `apslf_start_enabled` off: none of Sparlectra's start-value improvers.
- `qlimits_enabled=false`, `enable_pq_gen_controllers=false` (import): no
  reactive limits and no P(U)/Q(U) controllers on PQ generators (with them,
  each generator is clamped to its limits).
- `distributed_slack_enabled=false`: single slack.
- `wrong_branch_detection=:off`: a plausibility check on the solved
  voltages, which could reject a converged solution the oracle should judge.
- `tol=TOLERANCE_PU`: Sparlectra stops when the infinity norm of the
  mismatch vector (P and Q in p.u. at PQ buses, P and |V|-Vset at PV buses)
  is at most `tol`. The loop counts mismatch evaluations, so it gets
  `MAX_ITERATIONS + 1` for `MAX_ITERATIONS` Newton steps.
- `linear_solver=:umfpack_reuse` and `rectangular_workspace_reuse=true`
  (defaults): symbolic LU analysis and work arrays reused across iterations.
  Each solve rebuilds Ybus and computes the final injections and mismatch
  diagnostics: Sparlectra has no call that solves on a prebuilt Ybus, so
  these are timed as part of `solve`.
- Julia runs single-threaded (`PYTHON_JULIACALL_THREADS=1`); `verbose=0`,
  info and warning logs off.

Known losses, reported by the oracle rather than hidden:
- Remote voltage regulation (CGMES): Sparlectra solves it only as an outer
  loop (`machine_control=true`: the machine turns PQ and a secant controller
  moves its Q until the remote bus is within a deadband, clamped to Q
  limits). That breaks the no-outer-loop and no-limit rules, so it is off, and
  such a machine is held PV at its own bus at its start voltage (1 p.u.
  without SV) instead: a different problem wherever a case regulates
  remotely. cgmes_microgrid_be: 2.6% median deviation from SV (pypowsybl,
  which regulates remotely: 0.48%; pypowsybl with it off measured 2.5%).
- Tabular phase tap changer (cgmes_powerflow): 2e-5 p.u. and 0.003 deg from
  SV where pypowsybl and pandapower reach 1e-6. The importer moves the tap to
  the other winding as 1/ratio at -angle; solving the 2-bus fixture by hand,
  the SV is reproduced to 1.4e-6 with the series impedance on end 1 and
  moved to 2.1e-5 when it is referred to the wrong side of the tap, so the
  impedance is probably not re-referred (not confirmed in the source).

Robustness (not a loss: the model is exact):
- From the common flat start, the rectangular Newton-Raphson diverges on
  case9241pegase (mismatch 530, 1.1e4, 1.4e6 p.u. after two steps) although
  MATPOWER's own solution satisfies Sparlectra's equations to 4e-10 MW, and
  the polar Newton-Raphson of MATPOWER and four other tools converges from
  that start. With PV buses started at 1 p.u. instead of their setpoints it
  converges in 8 steps. The same holds for case9241pegase@cimoxide and
  cgmes_realgrid, which converge (and are accepted) from a 1 p.u. start.
"""
from importlib.metadata import version

from adapters.cgmes_ids import by_node
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import CACHE, CASES, is_cgmes


def _gb():
    from adapters.sparlectra_julia import GB   # starts Julia on first use
    return GB


class SparlectraAdapter(SolverAdapter):
    name = "sparlectra"
    display_name = "Sparlectra.jl"
    color = "#0f8fa8"
    package = "juliacall"
    modules = ("adapters.sparlectra_julia",)
    language = "julia"
    families = ("matpower", "distribution", "cgmes", "converted-cimoxide", "converted-pypowsybl")
    settings = {"solver": "runpf_rectangular!", "formulation": "rectangular", "init": "flat", "damping": 1.0,
                "start_improvers": False, "reactive_limits": False, "distributed_slack": False,
                "remote_voltage_control": False, "outer_loop_controls": "off", "tolerance_pu": TOLERANCE_PU,
                "max_iteration": MAX_ITERATIONS}

    def version(self):
        return self.dependencies()["Sparlectra"]

    def dependencies(self):
        from adapters.sparlectra_julia import JULIA_VERSION
        return {k: str(v) for k, v in _gb().versions().items()} | {"julia": JULIA_VERSION,
                                                                    "juliacall": version("juliacall")}

    def load(self, case):
        if is_cgmes(case):
            return _gb().load_cgmes(str(CACHE / f"{case}.zip"))
        return _gb().load_matpower(str(CASES[case]["file"]))

    def solve(self, model):
        steps = _gb().solve_b(model, TOLERANCE_PU, MAX_ITERATIONS)   # juliacall spells solve! as solve_b
        if steps < 0:
            raise DidNotConverge(f"NR did not converge in {MAX_ITERATIONS} iterations")
        self._iterations = int(steps)

    def solution(self, model, case):
        ids, vm, va = _gb().solution(model)
        ids = [str(i) for i in ids]
        vm, va = dict(zip(ids, map(float, vm))), dict(zip(ids, map(float, va)))
        if is_cgmes(case):
            vm, va = by_node(case, {i: (vm[i], va[i]) for i in ids}, "node")
        return Solution(vm, va, self._iterations)
