"""The interface every DC power-flow tool implements (problems "dc" and
"dcn1", `oracle.dc`).

The problem is MATPOWER's DC model (`rundcpf`, `makeBdc`): branch admittance
1 / (x * tap), resistance and charging left out, phase shifts as injections,
shunt conductance as load, one reference bus. The same three calls as a
power-flow adapter: `load` (timed as `import`), `solve` (timed), `solution`,
which returns bus angles (degrees, by MATPOWER bus number) and from-end
active branch flows (MW, by row of the `.m`'s branch matrix): both are
graded. A DC solve has no start and no tolerance: it is one linear solve.

DC N-1 (`DcContingencyAdapter`, problem "dcn1"): the N-1 cases' outages
(`cases.contingency`, through `outages_of`), the base case and every outage
in one timed `solve`, with the batch machinery's modes (`native`, `loop`) and
thread counts (`adapters.batch_adapter`). An outage may be computed by a
new solve or from the base case's factors or sensitivities (LODF, PTDF):
both are exact in DC, so both are graded the same way, and `settings`
records which. A tool must report the angles of every outage too.
"""
from dataclasses import dataclass

import numpy as np

from adapters.batch_adapter import BatchAdapter
from adapters.solver_adapter import SolvingAdapter


@dataclass
class DcSolution:
    bus_ids: list[str] | None   # MATPOWER bus numbers; None: the tool computes no angles (oracle.dc)
    va_deg: np.ndarray | None   # per bus_ids (outages x bus_ids for DC N-1)
    branch_rows: list[int]    # rows of the .m's branch matrix
    pf_mw: np.ndarray         # per branch_rows (outages x branch_rows for DC N-1)


class DcAdapter(SolvingAdapter):
    problem = "dc"
    families = ("dc-matpower",)

    def tags(self) -> list[str]:
        return ["powerflow", "dc", self.language]


class DcContingencyAdapter(BatchAdapter):
    problem = "dcn1"
    families = ("dcn1-matpower",)

    def tags(self) -> list[str]:
        return ["powerflow", "dc", "contingency", self.mode, self.language]
