"""Reads a CGMES bus-branch model back into MATPOWER terms, without any tool.

Used for MATPOWER cases converted to CGMES (`cases/matpower_to_cgmes.py`
with cimoxide, `cases/convert_pypowsybl.py` with pypowsybl). Two uses:

1. **Joining a tool's solution to the original case.** Every converted
   TopologicalNode is named `BUS-<n>`, the MATPOWER bus number; its
   BaseVoltage turns a tool's kV back into per unit. With that, the tier-1
   residual against the original `.m` grades tools on converted input too.
2. **Converter fidelity** (`fidelity`). Rebuilds Ybus, the specified bus
   injections and the voltage setpoints from the CGMES files and compares
   them with the `.m`. This grades the converter alone, before any tool
   reads its output.

This parser is ElementTree only and shares no code with cimoxide, so that
the cimoxide-based converter is not checked by its own library. pypowsybl's
export is checked by the same code, which guards against this reader
misreading CGMES conventions.

CGMES conventions assumed (IEC 61970-301 / CGMES 3.0), and exercised by both
converters:
- ACLineSegment: pi-model, r/x in ohms, `bch`/`gch` total, split half to
  each end.
- PowerTransformerEnd: r/x of each end referred to that end's `ratedU`; the
  series impedances of both ends add up (end 1 transferred through the ideal
  ratio). Each end's shunt b/g sits at that end's terminal, in siemens at the
  terminal's voltage, the reading PowSyBl's conversion
  (`b = b1 * (ratedU1/ratedU2)^2 + b2`) is consistent with. The ideal
  ratio is ratedU1/ratedU2 relative to the nodes' nominal voltages, times
  any tap changer ratio, and a phase tap changer shifts end 1's angle.
- Injections in load sign convention (SSH `p`/`q` positive = consumption);
  SynchronousMachine voltage control through its RegulatingControl.
"""
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
BUS_NAME = re.compile(r"^BUS-(\d+)$")


def read(paths: list[Path]) -> dict[str, dict]:
    """{mrid: {"_class": ..., "<Class.property>": value-or-ref}} merged over
    all profiles (rdf:ID and rdf:about "#_x" name the same object)."""
    objs: dict[str, dict] = defaultdict(dict)
    for path in paths:
        for el in ET.parse(path).getroot():
            ident = el.get(f"{RDF}ID") or el.get(f"{RDF}about", "").lstrip("#")
            if not ident or ident.startswith("urn:"):
                continue
            obj = objs[ident.lstrip("_")]
            obj.setdefault("_class", el.tag.split("}")[1])
            for child in el:
                prop = child.tag.split("}")[1]
                ref = child.get(f"{RDF}resource")
                obj[prop] = ref.lstrip("#").lstrip("_") if ref is not None else child.text
    return dict(objs)


def of_class(objs: dict, *classes: str) -> dict[str, dict]:
    return {k: v for k, v in objs.items() if v["_class"] in classes}


def _f(obj: dict, prop: str, default: float = 0.0) -> float:
    v = obj.get(prop)
    return default if v is None or v == "" else float(v)


def _true(obj: dict, prop: str, default: bool = True) -> bool:
    v = obj.get(prop)
    return default if v is None else v.strip().lower() == "true"


def node_buses(objs: dict) -> dict[str, tuple[int, float]]:
    """{topological node mrid: (MATPOWER bus number, nominal kV)}."""
    out = {}
    for m, tn in of_class(objs, "TopologicalNode").items():
        match = BUS_NAME.match(tn.get("IdentifiedObject.name", ""))
        if match:
            kv = _f(objs[tn["TopologicalNode.BaseVoltage"]], "BaseVoltage.nominalVoltage")
            out[m] = (int(match.group(1)), kv)
    return out


def to_matpower_ids(objs: dict, vm_kv: dict[str, float], va_deg: dict[str, float]) -> tuple[dict, dict]:
    """A tool's solution keyed by TopologicalNode -> keyed by MATPOWER bus, in p.u."""
    vm, va = {}, {}
    for tn, (bus, kv) in node_buses(objs).items():
        key = tn.replace("-", "").lower()
        if key in vm_kv:
            vm[str(bus)], va[str(bus)] = vm_kv[key] / kv, va_deg[key]
    return vm, va


def network(objs: dict, s_base: float) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray, np.ndarray]:
    """(bus_ids, Ybus in p.u., specified injection in p.u., |V| setpoint in p.u.
    (NaN where no enabled voltage control)) rebuilt from the CGMES objects."""
    buses = node_buses(objs)
    ids = np.array(sorted(b for b, _ in buses.values()))
    pos = {b: i for i, b in enumerate(ids)}
    node_pos = {tn: pos[b] for tn, (b, _) in buses.items()}
    node_kv = {tn: kv for tn, (_, kv) in buses.items()}
    terms = of_class(objs, "Terminal")
    by_equipment = defaultdict(dict)
    for t in terms.values():
        if _true(t, "ACDCTerminal.connected"):
            by_equipment[t["Terminal.ConductingEquipment"]][int(_f(t, "ACDCTerminal.sequenceNumber", 1))] = t
    n = len(ids)
    rows, cols, vals = [], [], []

    def stamp(f, t, yff, yft, ytf, ytt):
        rows.extend([f, f, t, t]); cols.extend([f, t, f, t]); vals.extend([yff, yft, ytf, ytt])

    for m, line in of_class(objs, "ACLineSegment").items():
        ends = by_equipment.get(m, {})
        if len(ends) < 2 or not _true(line, "Equipment.inService"):
            continue
        f, t = ends[1]["Terminal.TopologicalNode"], ends[2]["Terminal.TopologicalNode"]
        z_base = node_kv[f] ** 2 / s_base
        ys = 1 / complex(_f(line, "ACLineSegment.r"), _f(line, "ACLineSegment.x")) * z_base
        ysh = complex(_f(line, "ACLineSegment.gch"), _f(line, "ACLineSegment.bch")) * z_base / 2
        stamp(node_pos[f], node_pos[t], ys + ysh, -ys, -ys, ys + ysh)

    ends_of = defaultdict(dict)
    for m, end in of_class(objs, "PowerTransformerEnd").items():
        ends_of[end["PowerTransformerEnd.PowerTransformer"]][int(_f(end, "TransformerEnd.endNumber"))] = (m, end)
    ptc_of = {}
    for m, ptc in of_class(objs, "PhaseTapChangerTabular", "PhaseTapChangerLinear").items():
        ptc_of[ptc["TapChanger.TransformerEnd"] if "TapChanger.TransformerEnd" in ptc
               else ptc["PhaseTapChanger.TransformerEnd"]] = (m, ptc)
    points = defaultdict(dict)
    for p in of_class(objs, "PhaseTapChangerTablePoint").values():
        points[p["PhaseTapChangerTablePoint.PhaseTapChangerTable"]][int(_f(p, "TapChangerTablePoint.step"))] = p
    rtc_of = {rtc.get("RatioTapChanger.TransformerEnd"): rtc for rtc in of_class(objs, "RatioTapChanger").values()}

    for m, xf in of_class(objs, "PowerTransformer").items():
        if not _true(xf, "Equipment.inService") or len(ends_of[m]) != 2:
            continue
        (m1, e1), (m2, e2) = ends_of[m][1], ends_of[m][2]
        n1 = terms[e1["TransformerEnd.Terminal"]]["Terminal.TopologicalNode"]
        n2 = terms[e2["TransformerEnd.Terminal"]]["Terminal.TopologicalNode"]
        if not (_true(terms[e1["TransformerEnd.Terminal"]], "ACDCTerminal.connected")
                and _true(terms[e2["TransformerEnd.Terminal"]], "ACDCTerminal.connected")):
            continue
        u1, u2 = _f(e1, "PowerTransformerEnd.ratedU"), _f(e2, "PowerTransformerEnd.ratedU")
        ratio = (u1 / node_kv[n1]) / (u2 / node_kv[n2])
        angle = 0.0
        for em in (m1, m2):
            if em in rtc_of:
                rtc = rtc_of[em]
                step = _f(rtc, "TapChanger.step") - _f(rtc, "TapChanger.neutralStep")
                k = 1 + step * _f(rtc, "RatioTapChanger.stepVoltageIncrement") / 100
                ratio = ratio * k if em == m1 else ratio / k
        if m1 in ptc_of:
            pm, ptc = ptc_of[m1]
            if ptc["_class"] == "PhaseTapChangerTabular":
                pt = points[ptc["PhaseTapChangerTabular.PhaseTapChangerTable"]][int(_f(ptc, "TapChanger.step"))]
                angle = _f(pt, "PhaseTapChangerTablePoint.angle")
                ratio *= _f(pt, "TapChangerTablePoint.ratio", 1.0)
            else:
                angle = (_f(ptc, "TapChanger.step") - _f(ptc, "TapChanger.neutralStep")) * \
                    _f(ptc, "PhaseTapChangerLinear.stepPhaseShiftIncrement")
        # Series impedance referred to end 2, in p.u. of node 2's base.
        z2_base = node_kv[n2] ** 2 / s_base
        z = (complex(_f(e1, "PowerTransformerEnd.r"), _f(e1, "PowerTransformerEnd.x")) * (u2 / u1) ** 2
             + complex(_f(e2, "PowerTransformerEnd.r"), _f(e2, "PowerTransformerEnd.x"))) / z2_base
        ys = 1 / z
        y1 = complex(_f(e1, "PowerTransformerEnd.g"), _f(e1, "PowerTransformerEnd.b")) * node_kv[n1] ** 2 / s_base
        y2 = complex(_f(e2, "PowerTransformerEnd.g"), _f(e2, "PowerTransformerEnd.b")) * node_kv[n2] ** 2 / s_base
        t = ratio * np.exp(1j * np.deg2rad(angle))
        f_, t_ = node_pos[n1], node_pos[n2]
        stamp(f_, t_, ys / (t * np.conj(t)) + y1, -ys / np.conj(t), -ys / t, ys + y2)

    s = np.zeros(n, complex)
    vset = np.full(n, np.nan)
    for m, sh in of_class(objs, "LinearShuntCompensator").items():
        for term in by_equipment.get(m, {}).values():
            if _true(sh, "Equipment.inService"):
                i, kv = node_pos[term["Terminal.TopologicalNode"]], node_kv[term["Terminal.TopologicalNode"]]
                sec = _f(sh, "ShuntCompensator.sections", 1.0)
                y = complex(_f(sh, "LinearShuntCompensator.gPerSection"), _f(sh, "LinearShuntCompensator.bPerSection"))
                rows.append(i); cols.append(i); vals.append(y * sec * kv ** 2 / s_base)
    for m, obj in of_class(objs, "EnergyConsumer", "ConformLoad", "NonConformLoad", "EnergySource",
                            "SynchronousMachine").items():
        if not _true(obj, "Equipment.inService"):
            continue
        for term in by_equipment.get(m, {}).values():
            i = node_pos[term["Terminal.TopologicalNode"]]
            if obj["_class"] == "EnergySource":   # also load sign convention (CIM: "positive = flow out")
                s[i] -= complex(_f(obj, "EnergySource.activePower"), _f(obj, "EnergySource.reactivePower")) / s_base
            else:
                cls = "EnergyConsumer" if obj["_class"] != "SynchronousMachine" else "RotatingMachine"
                s[i] -= complex(_f(obj, f"{cls}.p"), _f(obj, f"{cls}.q")) / s_base
            rc = objs.get(obj.get("RegulatingCondEq.RegulatingControl", ""))
            if (obj["_class"] == "SynchronousMachine" and rc and _true(obj, "RegulatingCondEq.controlEnabled", False)
                    and _true(rc, "RegulatingControl.enabled", False)):
                target = terms[rc["RegulatingControl.Terminal"]]["Terminal.TopologicalNode"]
                vset[node_pos[target]] = _f(rc, "RegulatingControl.targetValue") / node_kv[target]
    y = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return ids, y, s, vset


def fidelity(mpc: dict, paths: list[Path]) -> dict:
    """How exactly a converted CGMES case represents its MATPOWER source:
    largest element-wise differences of Ybus (p.u.), specified injections
    (MVA) and voltage setpoints (p.u.), plus bus counts. Zero (to rounding)
    means every tool reading the files faces the same problem as the `.m`."""
    from cases.matpower import PV, REF
    from oracle.residual import effective_bus_types, specified_injections
    from oracle.ybus import make_ybus

    objs = read(paths)
    s_base = float(mpc["baseMVA"])
    ids_c, y_c, s_c, v_c = network(objs, s_base)
    ids_m, y_m = make_ybus(mpc)
    s_m, v_m = specified_injections(mpc, ids_m)
    # Only buses MATPOWER solves as PV or slack have a setpoint (bustypes.m).
    v_m = np.where(np.isin(effective_bus_types(mpc, v_m), (PV, REF)), v_m, np.nan)
    keep = np.isin(ids_m, ids_c)                   # isolated buses are not converted
    ids_m, y_m, s_m, v_m = ids_m[keep], y_m[keep][:, keep], s_m[keep], v_m[keep]
    order = np.searchsorted(ids_c, ids_m)
    y_c, s_c, v_c = y_c[order][:, order], s_c[order], v_c[order]
    dy = abs(y_c - y_m)
    both = ~np.isnan(v_m) & ~np.isnan(v_c)
    return {
        "buses": int(len(ids_c)), "buses_expected": int(len(ids_m)),
        "max_dy_pu": float(dy.max()) if dy.nnz else 0.0,
        "max_dy_rel": float(dy.max() / abs(y_m).max()) if dy.nnz else 0.0,
        "max_ds_mva": float(np.abs(s_c - s_m).max() * s_base),
        "setpoints_missing": int((~np.isnan(v_m) & np.isnan(v_c)).sum()),
        "setpoints_extra": int((np.isnan(v_m) & ~np.isnan(v_c)).sum()),
        "max_dv_setpoint_pu": float(np.abs(v_c - v_m)[both].max()) if both.any() else 0.0,
    }
