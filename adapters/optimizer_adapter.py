"""The interface every AC optimal-power-flow tool implements.

The same three calls as a power-flow adapter (`adapters.solver_adapter`),
timed the same way: `load` reads the case (timed as `import`), `solve` runs
the OPF repeatedly on that one persistent model, `solution` returns bus
voltages and the dispatch of every online generator (`Solution.pg_mw`,
`qg_mvar`, keyed by gen row) for the oracle (`oracle.opf`).

Every adapter must configure its tool to solve the same problem as every
other, MATPOWER's AC-OPF in polar form:

- minimise the polynomial generator cost of the case (`gencost` model 2);
- subject to the AC power-flow equations, bus voltage limits, generator P
  and Q limits, apparent-power limits at both ends of each branch (`rateA`;
  0 unlimited) and branch angle-difference limits (`angmin`/`angmax`);
- no discrete controls, no distributed slack, no DC approximation;
- flat start where the tool lets one be set (every bus at 1 p.u. and 0
  degrees, generators at the middle of their P and Q ranges), otherwise the
  tool's default, stated;
- `OPF_TOLERANCE` on feasibility, gradient and complementarity where the
  tool exposes them, `OPF_MAX_ITERATIONS`.

A constraint a tool models differently (a current limit where the case
gives an MVA limit, no angle-difference limits) is stated in its docstring
and left to the oracle to show, never patched. Each adapter's docstring
justifies each setting, gives the bus and generator joins (MATPOWER bus
number, gen row), and says what it deliberately does not do.

A tool keeps its power-flow adapter's identity (`name`, `color`, ...); its
OPF results go to `<tool>-opf.json`.
"""
from adapters.solver_adapter import ToolAdapter

OPF_TOLERANCE = 1e-6   # MATPOWER's default for MIPS; interior-point tolerances are relative
OPF_MAX_ITERATIONS = 200


class OptimizerAdapter(ToolAdapter):
    problem = "opf"
    families = ("opf-pglib",)

    def tags(self) -> list[str]:
        return ["opf", "ac", "interior-point", self.language]
