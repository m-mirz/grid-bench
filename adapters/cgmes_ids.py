"""Maps each tool's CGMES buses onto TopologicalNode mRIDs, for adapters.

Kept out of `oracle/` deliberately: this needs tool objects, the oracle must
not. The join itself (terminal or connectivity node -> TopologicalNode) comes
from `oracle.cgmes_sv.topology`, i.e. from the case's own TP profile.
"""
from functools import lru_cache

from cases.registry import cgmes_tp_files
from oracle.cgmes_sv import mrid, topology


@lru_cache(maxsize=None)
def tp_maps(case: str) -> tuple[dict[str, str], dict[str, str]]:
    return topology(cgmes_tp_files(case))


def by_node(case: str, per_bus: dict[str, tuple[float, float]], key: str) -> tuple[dict, dict]:
    """`per_bus` maps a tool's element (a terminal or connectivity node mRID,
    as selected by `key`) to (v, angle_deg); returns (vm, va) keyed by TN."""
    terminals, cns = tp_maps(case)
    lookup = terminals if key == "terminal" else cns
    nodes = set(terminals.values()) | set(cns.values())
    vm, va = {}, {}
    for elem, (v, a) in per_bus.items():
        # A bus-branch model has no ConnectivityNodes; a tool's bus is then
        # the TopologicalNode itself.
        tn = lookup.get(mrid(elem)) or (mrid(elem) if mrid(elem) in nodes else None)
        if tn is not None:
            vm[tn], va[tn] = v, a
    return vm, va
