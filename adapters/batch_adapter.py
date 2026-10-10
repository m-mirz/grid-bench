"""The interface every batch power-flow tool implements.

A batch case (`cases.sweep`) is one network and many operating points. The
same three calls as a power-flow adapter (`adapters.solver_adapter`): `load`
reads the network and the scenarios (timed as `import`), `solve` solves
every scenario in one call, on `threads` threads (timed as `solve`, once per
thread count), `solution` returns every scenario's voltages for the oracle
(`oracle.batch`).

Each scenario is the problem a power-flow adapter solves (a single slack,
no reactive limits, no outer-loop controls, `TOLERANCE_PU`,
`MAX_ITERATIONS`), started from the tool's own solution of the base case
(the case's own demand and dispatch), as N-1 starts its outages: one timed
`solve` is the base-case power flow from flat start and every scenario,
each from that solution. That is how a sweep is run from a solved base case
(a Monte-Carlo study, an operating-point scan), and every scenario still
starts from the same point, never from another scenario's result (which
most time-series APIs default to), so scenarios stay independent and
threads cannot share state through them. A tool that cannot start a
scenario from the base case says so in its docstring and what it does
instead. What changes between
scenarios is demand (`Pd`, `Qd` per bus) and dispatch (`Pg` per generator);
an adapter updates exactly the elements its importer created for those,
joined by MATPOWER bus number and gen row, never by position.

Two modes, recorded in `settings["mode"]`:

- `native`: the tool's own batch API, given the scenarios as data. Only
  these are timed at more than one thread count (`thread_counts`), if the
  API has a thread setting (`threaded`), with that setting; BLAS stays single-threaded (the run scripts
  set `OPENBLAS_NUM_THREADS=1`), so the parallelism measured is the tool's.
- `loop`: a tool with no batch API, or none that states this problem, solves
  the base case on its power-flow adapter's model, then every scenario
  written into that model and started from the base solution
  (`solve_base`, `solve_scenario`), on one thread. That is how such a tool
  is used for a sweep, and it is the reference the native batch APIs are
  compared against: the time per scenario of a loop is close to a single
  solve from a near start, plus the cost of the update.

Reading the voltages of every scenario is part of the timed call: a sweep
that keeps only the last result is not a sweep.

A tool keeps its power-flow adapter's identity (`name`, `color`, ...); its
batch results go to `<tool>-batch.json`.

N-1 contingency analysis (`ContingencyAdapter`, problem "n1") is the same
machinery on a case's outages, each outage starting from the base case's
solution the same way (see its docstring).
"""
import os
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from adapters.solver_adapter import SolvingAdapter, ToolAdapter
from cases.contingency import read as read_outages
from cases.registry import CASES, contingency_path, sweep_path
from cases.sweep import read


@dataclass
class BatchSolution:
    """Voltages of every scenario, columns keyed by MATPOWER bus number (as
    str), rows in scenario order."""
    bus_ids: list[str]
    vm: np.ndarray        # scenarios x buses, p.u.
    va_deg: np.ndarray


def available_threads() -> int:
    """Cores this process may run on (the container's cpuset), not the host's."""
    return len(os.sched_getaffinity(0))


def thread_counts() -> tuple[int, ...]:
    """1, 2, 4, ... and every available core."""
    n = available_threads()
    return tuple(sorted({2 ** i for i in range(n.bit_length()) if 2 ** i <= n} | {n}))


def base_case(case: str) -> str:
    return CASES[case]["base_case"]


def scenarios(case: str) -> dict[str, np.ndarray]:
    return read(sweep_path(case))


def columns(ids, keys) -> np.ndarray:
    """Position of each of `ids` (MATPOWER bus numbers) in `keys`, the
    scenario data's column ids; every one must exist (rule 5)."""
    pos = {int(k): i for i, k in enumerate(keys)}
    missing = [int(i) for i in ids if int(i) not in pos]
    assert not missing, f"no scenario column for {missing[:5]}"
    return np.array([pos[int(i)] for i in ids], dtype=np.int64)


class BatchAdapter(SolvingAdapter):
    problem = "batch"
    families = ("sweep-matpower", "sweep-distribution")
    mode = "native"

    def tags(self) -> list[str]:
        return ["powerflow", "ac", "batch", self.mode, self.language]

    threaded = True   # the batch API takes a thread count

    def thread_counts(self) -> tuple[int, ...]:
        return thread_counts() if self.mode == "native" and self.threaded else (1,)

    @abstractmethod
    def solve(self, model, threads: int = 1) -> None:
        """Every scenario. Raises DidNotConverge if any does not converge."""


class LoopBatchAdapter(BatchAdapter):
    """A loop over the power-flow adapter's model: solve the base case, then
    for every scenario write it into the model, solve from the base
    solution, keep the voltages. Subclasses set `single` and implement the
    hooks; joins belong in `load`."""
    mode = "loop"
    single: ToolAdapter

    @abstractmethod
    def solve_base(self, model) -> None:
        """Restores the case's own demand and dispatch and solves it from
        flat start; keeps what the scenarios start from. Raises DidNotConverge."""

    @abstractmethod
    def solve_scenario(self, model, k: int) -> None:
        """Writes scenario k's demand and dispatch into the model and solves
        it from the base solution. Raises DidNotConverge."""

    @abstractmethod
    def voltages(self, model) -> tuple[np.ndarray, np.ndarray]:
        """|V| and angle (degrees) of the last solve, in `model["bus_ids"]` order."""

    def solve(self, model, threads: int = 1) -> None:
        assert threads == 1, "a loop runs on one thread"
        n = len(model["sweep"]["scale"])
        vm, va = np.empty((n, len(model["bus_ids"]))), np.empty((n, len(model["bus_ids"])))
        self.solve_base(model)
        for k in range(n):
            self.solve_scenario(model, k)
            vm[k], va[k] = self.voltages(model)
        model["vm"], model["va"] = vm, va

    def solution(self, model, case) -> BatchSolution:
        return BatchSolution(model["bus_ids"], model["vm"], model["va"])


def outages(case: str) -> dict[str, np.ndarray]:
    return read_outages(contingency_path(case))


class ContingencyAdapter(BatchAdapter):
    """N-1 contingency analysis (`cases.contingency`): the same calls and
    modes as a batch, on the outages of a case. One timed `solve` is the
    base-case power flow (flat start, as everywhere) and every outage, each
    started from the tool's own solution of the base case: the one exception
    to the flat start, as contingency analysis is done. A tool that cannot
    start an outage from the base case says so in its docstring and what it
    does instead. `solution` returns the outages' voltages, in outage order.
    Results go to `<tool>-n1.json`."""
    problem = "n1"
    families = ("n1-matpower",)

    def tags(self) -> list[str]:
        return ["powerflow", "ac", "contingency", self.mode, self.language]


class LoopContingencyAdapter(ContingencyAdapter):
    """A loop over the power-flow adapter's model: solve the base case, then
    for every outage take its branch out, solve from the base solution,
    keep the voltages, put the branch back. Subclasses implement the hooks."""
    mode = "loop"

    @abstractmethod
    def solve_base(self, model) -> None:
        """The base case from flat start; keeps what the outages start from.
        Raises DidNotConverge."""

    @abstractmethod
    def solve_outage(self, model, k: int) -> None:
        """Outage k from the base solution, leaving the model as it was
        except for the result. Raises DidNotConverge."""

    @abstractmethod
    def voltages(self, model) -> tuple[np.ndarray, np.ndarray]:
        """|V| and angle (degrees) of the last solve, in `model["bus_ids"]` order."""

    def solve(self, model, threads: int = 1) -> None:
        assert threads == 1, "a loop runs on one thread"
        n = len(model["outages"]["branch_row"])
        vm, va = np.empty((n, len(model["bus_ids"]))), np.empty((n, len(model["bus_ids"])))
        self.solve_base(model)
        for k in range(n):
            self.solve_outage(model, k)
            vm[k], va[k] = self.voltages(model)
        model["vm"], model["va"] = vm, va

    def solution(self, model, case) -> BatchSolution:
        return BatchSolution(model["bus_ids"], model["vm"], model["va"])
