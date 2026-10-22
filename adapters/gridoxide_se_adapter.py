"""gridoxide: `StateEstimationModel`, Gauss-Newton on the normal equations
(`newton_raphson`), in Rust, driven through its Python bindings.

Input: the document power-grid-model's estimator is given
(`adapters/pgm_se_adapter.py`): the PGM JSON of the power-flow converter
(`cases.prep`) without its `voltage_regulator`s, plus the case's measurement
set as `sym_voltage_sensor`s and `sym_power_sensor`s, joined the same way
(node = MATPOWER bus number; branch = the converter's id for that `.m` row,
from `<case>.pgm.branch-ids.json`, asserted against its from and to node;
`node` terminal for injections, `branch_from` for flows). It is built here
as plain JSON, not through power-grid-model, which this image does not
have. gridoxide reads PGM documents only from a file, so the document is
written once per case, on the first `load` (the untimed warm-up), into the
temporary directory (the checkout is read-only in the container), named by
the case and the modification times of what it is built from; `import`
times gridoxide reading it.

Sources: `ideal_sources=True`. The converter writes MATPOWER's reference
bus as a `source` of 1e40 VA, its stand-in for an ideal one, which
power-grid-model's own model reads as a virtual slack bus behind ~1e-32 p.u.
of impedance; gridoxide's estimator, which estimates that bus's magnitude like
any other, finds the gain matrix numerically singular there and converges on
no case. As an ideal source the reference bus is the slack itself, as in the
`.m`: its angle the reference, its magnitude estimated, its injection what
its sensors say.

Settings:
- `method="newton_raphson"`: WLS Gauss-Newton on the nonlinear measurement
  functions, as power-grid-model's. `iterative_linear` is a different
  estimator.
- `zero_injection=False`: no node is held at zero injection. By default
  gridoxide reads the document by power-grid-model's rules and constrains
  every node without a load or generator to inject exactly nothing,
  overriding its measurement, which the rules exclude (case14~noisy: J 42.08,
  power-grid-model's 42.1, against the optimum's 41.4). Off, such a node
  injects what its measurements say. gridoxide keeps the constraints its own
  model needs: a three-winding star point's, and, with impedance sources, a
  source node's (not here: an ideal source's node injects).
- `tol=SE_TOLERANCE` on the largest state update, `max_iter=MAX_ITERATIONS`.
- `backend`: the power-flow adapter's (`GRIDOXIDE_BACKEND`, default `klu`).
  The estimator keeps its symbolic factorization between solves.
- `s_base_va` = the case's `baseMVA`: per-unit as the case states it.
- Start (hardcoded, not settable from Python), on every `solve()`: of a
  linear solve of the network and the flat state, both with every measured
  |V| applied, whichever the measurements fit better; not always flat.

It inherits the converter's losses: transformer line charging moved into
end-node shunts, phase shifts rounded (see the power-flow and PGM adapters).

Iterations are not reported: the Python model does not expose them for a
single solve.

Results (first run, on the host rather than in the container; gridoxide
branch `se-node-injection`, uncommitted): J equals JuliaGrid's on every case
but the PEGASE ones and case300~noisy (884 against 881), which are the
converter's losses above; the oracle passes on all the others, `~noisy`
included. Not estimated: case3120sp~noisy and case9241pegase~noisy (no
convergence in 30 steps, as power-grid-model; JuliaGrid converges on both).
Solve time: within about 1.5x of JuliaGrid's either way on most cases
(case9241pegase~exact 0.50 s against 0.77 s), slower on mvlv10616~noisy
(122 ms against 88 ms). With the zero-injection constraints on, gridoxide
factors `[HᵀWH Cᵀ; C 0]`, which KLU orders badly: case9241pegase~exact took
2.9 s. With the source as power-grid-model models it (`sk` lowered to 1e10
VA to be estimable at all), case3120sp~exact reported a singular gain
matrix; as an ideal source it estimates exactly.

Fixed in gridoxide on the way: a node-injection sensor counted the bus's
shunt as injected (power-grid-model counts it as network); the power
derivative's leading-factor term was added once per coefficient on the
measured bus; the linear start, built from the document's power-flow
schedule, began case118~noisy at |V| = 0.67 p.u. and diverged (now the flat
state is a candidate too); `zero_injection` and `ideal_sources` are new.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from adapters.estimator_adapter import SE_TOLERANCE, EstimatorAdapter
from adapters.gridoxide_adapter import GridoxideAdapter
from adapters.solver_adapter import MAX_ITERATIONS, DidNotConverge, Solution
from cases import measurements
from cases.registry import CASES, measurements_path, pgm_branch_ids_path, pgm_json_path

WATT = 1e6   # MW -> W: the converter's powers are in W


class GridoxideEstimator(EstimatorAdapter):
    name = GridoxideAdapter.name
    display_name = GridoxideAdapter.display_name
    color = GridoxideAdapter.color
    package = GridoxideAdapter.package
    language = GridoxideAdapter.language
    modules = GridoxideAdapter.modules
    backend = GridoxideAdapter.backend
    settings = {"method": "newton_raphson", "backend": backend, "zero_injection": False,
                "init": "better of linear and flat (hardcoded)",
                "input": "PGM JSON with sensors", "ideal_sources": True, "tol": SE_TOLERANCE,
                "max_iter": MAX_ITERATIONS, "commit": os.environ.get("GRIDOXIDE_COMMIT", "unknown")}

    def load(self, case):
        import gridoxide
        path, base = document(case)
        model = gridoxide.StateEstimationModel.from_pgm_json(
            str(path), backend=self.backend, method="newton_raphson", tol=SE_TOLERANCE,
            max_iter=MAX_ITERATIONS, s_base_va=base * 1e6, zero_injection=False, ideal_sources=True)
        return {"model": model}

    def solve(self, model):
        try:
            model["model"].solve()
        except RuntimeError as e:
            raise DidNotConverge(str(e)) from e

    def solution(self, model, case):
        m = model["model"]
        vm, va = m.voltage_mag(), np.rad2deg(m.voltage_ang())
        # Node ids are MATPOWER bus numbers; the virtual source buses have none.
        nodes = m.nodes()
        return Solution({str(i): vm[k] for i, k in nodes}, {str(i): va[k] for i, k in nodes})


def document(case):
    """The sensor document of `case` on disk, written if there is none for
    the current inputs; and the case's base power, MVA."""
    base_case = CASES[case]["base_case"]
    sources = (pgm_json_path(base_case), pgm_branch_ids_path(base_case), measurements_path(case))
    stamp = hashlib.sha256(repr([s.stat().st_mtime_ns for s in sources]).encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / "grid-bench-gridoxide" / f"{case}.{stamp}.json"
    meas = measurements.read(measurements_path(case))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = json.loads(pgm_json_path(base_case).read_text())
        data = doc["data"]
        data.pop("voltage_regulator", None)
        branch_ids = {int(r): i for r, i in json.loads(pgm_branch_ids_path(base_case).read_text()).items()}
        ends = {b["id"]: (b["from_node"], b["to_node"]) for comp in ("line", "transformer") for b in data.get(comp, [])}
        u_rated = {n["id"]: n["u_rated"] for n in data["node"]}
        next_id = 1 + max(o["id"] for objs in data.values() for o in objs)

        volt, power = [], {}   # power: (object, terminal type) -> {"p": (value, sigma), "q": ...}
        for m in meas["measurements"]:
            if m["kind"] == "vm":
                volt.append({"measured_object": m["bus"], "u_measured": m["value"] * u_rated[m["bus"]],
                             "u_sigma": m["sigma"] * u_rated[m["bus"]]})
                continue
            if "bus" in m:
                key = (m["bus"], 9)   # MeasuredTerminalType.node
            else:
                obj = branch_ids[m["branch_row"]]
                assert ends[obj] == (m["from_bus"], m["to_bus"])
                key = (obj, 0)        # MeasuredTerminalType.branch_from
            power.setdefault(key, {})[m["kind"][0]] = (m["value"] * WATT, m["sigma"] * WATT)

        data["sym_voltage_sensor"] = [{"id": next_id + k, **v} for k, v in enumerate(volt)]
        next_id += len(volt)
        sensors = []
        for k, ((obj, terminal), pq) in enumerate(power.items()):
            assert set(pq) == {"p", "q"}   # cases.measurements always measures P and Q together
            sensors.append({"id": next_id + k, "measured_object": obj, "measured_terminal_type": terminal,
                            "p_measured": pq["p"][0], "p_sigma": pq["p"][1],
                            "q_measured": pq["q"][0], "q_sigma": pq["q"][1]})
        data["sym_power_sensor"] = sensors
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(doc))
        tmp.replace(path)
    return path, float(meas["baseMVA"])
