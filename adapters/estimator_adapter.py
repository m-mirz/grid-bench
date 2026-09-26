"""The interface every state estimator under test implements.

The same three calls as a power-flow adapter (`adapters.solver_adapter`),
timed the same way: `load` reads the network and attaches the case's
measurement set (`cases.measurements`, timed as `import`), `solve` runs the
estimator repeatedly on that one persistent model, `solution` returns the
estimated voltages for the oracle (`oracle.wls`).

Every adapter must configure its tool to solve the same problem as every
other, weighted least squares over the measurement set as given:

- plain WLS (Gauss-Newton on the normal equations, or the tool's closest
  equivalent), flat start on every solve, the slack bus angle as the only
  angle reference;
- every measurement with its own sigma, and nothing else: no
  pseudo-measurements, no zero-injection constraints, no measurements the
  tool derives on its own;
- no bad-data detection or removal, no topology or parameter estimation;
- convergence tolerance `SE_TOLERANCE` on the state update where the tool
  exposes one, `MAX_ITERATIONS`.

A measurement a tool cannot express raises `CaseUnsupported`; a setting a
tool hardcodes is documented in the adapter's docstring, never patched.
Each adapter's docstring justifies each setting, gives the bus and branch
joins (by MATPOWER bus number and branch row, never by position), and says
what it deliberately does not do.

A tool keeps its power-flow adapter's identity (`name`, `color`, ...); its
estimator results go to `<tool>-se.json`.

Tools without an estimator here: lightsim2grid, PyPSA, pypowsybl and Sienna
ship none. MATPOWER ships one (`extras/se`, `doSE`), but it cannot state
this problem: one sigma per measurement class, not per measurement;
injections only as generator outputs at generator buses, so a load bus's
injection cannot be measured; the slack's |V| held at its start value
instead of estimated; its tolerance hardcoded (1e-5 on the gradient norm).
Every case would be CaseUnsupported, so it has no adapter.
"""
from adapters.solver_adapter import ToolAdapter

SE_TOLERANCE = 1e-8   # max state update at convergence, p.u. and rad


class EstimatorAdapter(ToolAdapter):
    problem = "se"
    families = ("se-matpower", "se-distribution")

    def tags(self) -> list[str]:
        return ["state-estimation", "wls", self.language]
