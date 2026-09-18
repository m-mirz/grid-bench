"""Tier 2 of the oracle: a CGMES case's own published solution.

CGMES conformity configurations ship an SV (state variables) profile holding
`SvVoltage` per TopologicalNode. Each tool is compared to it independently,
so no tool-to-tool matching is needed.

Tools report voltages on their own bus objects. Each adapter maps those to
TopologicalNode mRIDs through a relationship in the TP profile
(`Terminal.TopologicalNode` or `ConnectivityNode.TopologicalNode`): an exact
join, never a nearest-voltage guess. (gridoxide's `cgmes_sv.py` matched
pypowsybl buses to nodes by nearest *published* voltage, which uses the
reference to decide what to compare against it.)

Caveats: the published SV comes from whatever solver and settings the
fixture's author used (controls, slack distribution), so it is a reference,
not ground truth. RealGrid's is known to be internally inconsistent.

Ported from gridoxide's `scripts/bench/cgmes_sv.py` (Apache-2.0, same author).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"


def mrid(ref: str) -> str:
    """Canonical form of an rdf:ID / rdf:resource / idtag: no '#', no leading
    '_', no dashes, lower case. VeraGrid drops the dashes; the rest keep them."""
    return ref.lstrip("#").lstrip("_").replace("-", "").lower()


def _iter(path: Path, tag: str):
    for _, el in ET.iterparse(path):
        if el.tag.endswith("}" + tag):
            yield el
            el.clear()


def _ref(el, prop: str) -> str | None:
    child = next((c for c in el if c.tag.endswith("}" + prop)), None)
    return None if child is None else mrid(child.get(f"{RDF}resource", ""))


def _text(el, prop: str) -> str | None:
    child = next((c for c in el if c.tag.endswith("}" + prop)), None)
    return None if child is None else child.text


def _id(el) -> str:
    return mrid(el.get(f"{RDF}ID") or el.get(f"{RDF}about", ""))


def published_voltages(sv_path: Path) -> dict[str, tuple[float, float]]:
    """{tn_mrid: (v_kv, angle_deg)} from the SV profile."""
    out = {}
    for sv in _iter(sv_path, "SvVoltage"):
        tn, v, a = _ref(sv, "SvVoltage.TopologicalNode"), _text(sv, "SvVoltage.v"), _text(sv, "SvVoltage.angle")
        if tn and v is not None and a is not None:
            out[tn] = (float(v), float(a))
    return out


def topology(tp_paths: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    """({terminal_mrid: tn_mrid}, {connectivity_node_mrid: tn_mrid}) from TP."""
    terminals, cns = {}, {}
    for path in tp_paths:
        for t in _iter(path, "Terminal"):
            tn = _ref(t, "Terminal.TopologicalNode")
            if tn:
                terminals[_id(t)] = tn
        for cn in _iter(path, "ConnectivityNode"):
            tn = _ref(cn, "ConnectivityNode.TopologicalNode")
            if tn:
                cns[_id(cn)] = tn
    return terminals, cns


def deviation(reference: dict[str, tuple[float, float]], vm_kv: dict[str, float],
              va_deg: dict[str, float]) -> dict:
    """Relative |V| error and angle error per shared node. Angles are compared
    after removing the median offset, since each tool picks its own angle
    reference. `n` is always reported: a small `n` means the mapping, not the
    solver, is what to look at."""
    shared = sorted(set(reference) & set(vm_kv))
    if not shared:
        return {"sv_n": 0, "sv_n_published": len(reference)}
    ref_v = np.array([reference[k][0] for k in shared])
    ref_a = np.array([reference[k][1] for k in shared])
    dv = np.abs(np.array([vm_kv[k] for k in shared]) - ref_v) / ref_v
    da = np.array([va_deg[k] for k in shared]) - ref_a
    da = np.abs((da - np.median(da) + 180) % 360 - 180)
    return {
        "sv_n": len(shared), "sv_n_published": len(reference),
        "sv_dv_median": float(np.median(dv)), "sv_dv_p90": float(np.quantile(dv, 0.9)), "sv_dv_max": float(dv.max()),
        "sv_da_max_deg": float(da.max()),
    }
