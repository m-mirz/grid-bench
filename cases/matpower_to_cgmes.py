"""MATPOWER case -> CGMES 3.0 (EQ, SSH, TP), written with cimoxide.

    python -m cases.matpower_to_cgmes <case.m> <out_dir>

One of two converters the benchmark compares (the other is pypowsybl's
MATPOWER import followed by its CGMES export, `cases/convert_pypowsybl.py`).
The goal is an *exact* representation of the MATPOWER power-flow problem, so
that the tier-1 residual against the original `.m` still applies to every
tool that reads the result. `oracle/cgmes_model.py` checks that claim without
any tool: it rebuilds Ybus, injections and setpoints from the written files.

Modelling decisions (MATPOWER manual, "Branch model" / `bustypes.m`):

- Physical units: each bus keeps its `baseKV` (0 -> 1 kV, see
  `normalize_for_tools`). Impedances in ohms, susceptances in siemens,
  powers in MW/MVAr at baseMVA.
- One VoltageLevel and one TopologicalNode per energized bus, bus-branch
  model (no ConnectivityNodes). TopologicalNode name is `BUS-<n>`, the same
  convention pypowsybl's export uses, which is how the oracle joins a tool's
  voltages back to MATPOWER bus numbers. Buses joined by transformers share a
  Substation.
- A branch with `ratio == 0` and `angle == 0` between buses of equal baseKV is
  an ACLineSegment (pi-model, `bch` = total line charging). Every other branch
  is a two-winding PowerTransformer:
  - MATPOWER's model is from bus - ideal transformer t:1 - series z - to bus,
    with b/2 at each end of z. The series z goes on end 2 (to side) in ohms of
    that side; both ends have b = g = 0.
  - The charging b becomes two LinearShuntCompensators: b/2 at the to bus,
    and (b/2)/|t|^2 at the from bus (MATPOWER's from-side b/2 sits behind the
    ideal transformer, which is that admittance seen from the from bus).
    This is exact, and it avoids transformer-end shunts, whose location tools
    interpret differently: PowSyBl collapses both ends' b onto one side, the
    very loss its own MATPOWER importer shows.
  - The off-nominal ratio is `ratedU1 = ratio * baseKV_from`,
    `ratedU2 = baseKV_to` (no RatioTapChanger).
  - A phase shift is a one-step PhaseTapChangerTabular on end 1
    (ratio 1, angle = MATPOWER `angle`).
- Loads: one EnergyConsumer per bus with Pd/Qd. Shunts: one
  LinearShuntCompensator per bus with Gs/Bs, one section in service.
- Generators: SynchronousMachine + GeneratingUnit, SSH p = -Pg, q = -Qg (load
  sign convention). Each has a voltage RegulatingControl at its own terminal
  with target Vg * baseKV. Regulation is enabled exactly where MATPOWER solves
  the bus as PV or slack: an online generator on a bus typed PV or REF. A
  generator on a PQ-typed bus is a fixed injection (control disabled), and an
  offline generator is out of service with its terminal disconnected.
  The slack is the REF bus generator, `referencePriority = 1`.
- Out-of-service branches and isolated buses are omitted: they do not enter
  the power-flow equations.
- Every line and transformer is in service (`Equipment.inService`, which
  cimoxide writes in SSH as the CGMES 3.0 `<cim:Equipment rdf:about>` form).
- Each profile gets a FullModel header naming grid-bench as modelling
  authority, with SSH and TP depending on EQ. (cimoxide >= 0.3.2 would
  synthesize a valid header itself; these carry the real values.)

Requires cimoxide >= 0.4 (pinned: 0.5.0), whose element dicts carry
`"Class.attr"` keys with string values (0.3.x took snake_case keys and typed
values). The files it writes are byte-identical to 0.3.3's on every case
converted by default. Versions before 0.3.2 wrote TopologicalNodes in TP as
bare references, dropped `Equipment.inService` for lines and transformers,
and synthesized headers PowSyBl ignores.
"""
import sys
import uuid
from pathlib import Path

import numpy as np

from cases.matpower import (ANGLE, BASE_KV, BR_B, BR_R, BR_STATUS, BR_X, BS, BUS_I, BUS_TYPE, F_BUS, GEN_BUS,
                            GEN_STATUS, GS, ISOLATED, PD, PG, PQ, PV, QD, QG, QMAX, QMIN, RATIO, REF, T_BUS, VG,
                            normalize_for_tools, parse_m)

NAMESPACE = uuid.UUID("6f1c2d3e-8a4b-4c5d-9e6f-7a8b9c0d1e2f")
PROFILES = {
    "EQ": "http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0",
    "SSH": "http://iec.ch/TC57/ns/CIM/SteadyStateHypothesis-EU/3.0",
    "TP": "http://iec.ch/TC57/ns/CIM/Topology-EU/3.0",
}
SCENARIO_TIME = "2026-01-01T00:00:00Z"
EMPTY = ('<?xml version="1.0" encoding="utf-8"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
         'xmlns:cim="http://iec.ch/TC57/CIM100#" xmlns:md="http://iec.ch/TC57/61970-552/ModelDescription/1#" '
         'xmlns:eu="http://iec.ch/TC57/CIM100-European#"></rdf:RDF>')


def text(value) -> str:
    """A value as RDF/XML writes it: booleans in lower case, numbers in their
    shortest round-trip form without exponent, as cimoxide < 0.4 wrote them,
    so the conversion stayed byte-identical across the API change."""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return np.format_float_positional(value, trim="-")
    return str(value)


class Builder:
    """Accumulates CIM objects keyed by deterministic UUIDs (uuid5 of a
    readable key), so the same case always converts to the same files.

    Fields are cimoxide's element dicts: `"Class.attr"` keys as in the XML
    (the class that declares the attribute, not the object's own), string
    values, a reference as the referenced id. cimoxide keeps an attribute
    its class does not declare, so a misspelt key is written, not dropped."""

    def __init__(self, case_name: str):
        self.case = case_name
        self.objects: dict[str, dict] = {}

    def add(self, cls: str, key: str, name: str | None, fields: dict) -> str:
        m = str(uuid.uuid5(NAMESPACE, f"{self.case}/{cls}/{key}"))
        # A table point is no IdentifiedObject in CGMES 3.0: no mRID, no name.
        identity = {} if name is None else {"IdentifiedObject.mRID": m, "IdentifiedObject.name": name}
        # An unbounded limit (MATPOWER's Qmax = Inf) is left out, as cimoxide
        # < 0.4 did: CGMES has no "unlimited", and `inf` is no xsd:float.
        self.objects[m] = {"_type": cls, "id": f"_{m}", **identity,
                           **{k: text(v) for k, v in fields.items() if not (isinstance(v, float) and np.isinf(v))}}
        return f"_{m}"

    def terminal(self, equipment: str, key: str, seq: int, node: str, connected: bool = True) -> str:
        return self.add("Terminal", f"{key}/T{seq}", f"{key} T{seq}", {
            "Terminal.ConductingEquipment": equipment, "ACDCTerminal.sequenceNumber": seq,
            "Terminal.TopologicalNode": node, "ACDCTerminal.connected": connected})


def _substations(mpc: dict, energized: np.ndarray) -> dict[int, int]:
    """Buses joined by any transformer share a Substation (union-find)."""
    parent = {int(b): int(b) for b in energized}

    def find(b):
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b

    kv = dict(zip(mpc["bus"][:, BUS_I].astype(int), mpc["bus"][:, BASE_KV]))
    for br in mpc["branch"][mpc["branch"][:, BR_STATUS] != 0]:
        f, t = int(br[F_BUS]), int(br[T_BUS])
        if _is_transformer(br, kv[f], kv[t]):
            parent[find(f)] = find(t)
    return {b: find(b) for b in parent}


def _is_transformer(br, kv_f: float, kv_t: float) -> bool:
    return br[RATIO] != 0 or br[ANGLE] != 0 or kv_f != kv_t


def convert(mpc: dict, case_name: str) -> Builder:
    mpc = normalize_for_tools(mpc)
    s_base = float(mpc["baseMVA"])
    bus, gen = mpc["bus"], mpc["gen"]
    energized = bus[bus[:, BUS_TYPE] != ISOLATED]
    kv = {int(b[BUS_I]): float(b[BASE_KV]) for b in energized}
    btype = {int(b[BUS_I]): int(b[BUS_TYPE]) for b in energized}
    b = Builder(case_name)

    region = b.add("GeographicalRegion", "region", case_name, {})
    subregion = b.add("SubGeographicalRegion", "subregion", case_name, {"SubGeographicalRegion.Region": region})
    base_voltages = {v: b.add("BaseVoltage", f"bv/{v:g}", f"{v:g} kV", {"BaseVoltage.nominalVoltage": v})
                     for v in sorted(set(kv.values()))}
    substation_of = _substations(mpc, energized[:, BUS_I])
    substations = {s: b.add("Substation", f"sub/{s}", f"SUB-{s}", {"Substation.Region": subregion})
                   for s in set(substation_of.values())}
    vl, tn = {}, {}
    for n in kv:
        vl[n] = b.add("VoltageLevel", f"vl/{n}", f"VL-{n}", {
            "VoltageLevel.BaseVoltage": base_voltages[kv[n]], "VoltageLevel.Substation": substations[substation_of[n]]})
        tn[n] = b.add("TopologicalNode", f"tn/{n}", f"BUS-{n}", {
            "TopologicalNode.BaseVoltage": base_voltages[kv[n]], "TopologicalNode.ConnectivityNodeContainer": vl[n]})

    for i, br in enumerate(mpc["branch"]):
        if br[BR_STATUS] == 0:
            continue
        f, t = int(br[F_BUS]), int(br[T_BUS])
        key = f"br/{i}"
        if not _is_transformer(br, kv[f], kv[t]):
            z_base = kv[f] ** 2 / s_base
            line = b.add("ACLineSegment", key, f"LINE-{f}-{t}-{i}", {
                "ConductingEquipment.BaseVoltage": base_voltages[kv[f]], "ACLineSegment.r": br[BR_R] * z_base,
                "ACLineSegment.x": br[BR_X] * z_base, "ACLineSegment.bch": br[BR_B] / z_base,
                "ACLineSegment.gch": 0.0, "Conductor.length": 1.0, "Equipment.inService": True})
            b.terminal(line, key, 1, tn[f])
            b.terminal(line, key, 2, tn[t])
            continue
        tap = br[RATIO] if br[RATIO] != 0 else 1.0
        z2 = kv[t] ** 2 / s_base
        xf = b.add("PowerTransformer", key, f"TWT-{f}-{t}-{i}", {
            "Equipment.EquipmentContainer": substations[substation_of[f]], "Equipment.inService": True})
        t1 = b.terminal(xf, key, 1, tn[f])
        t2 = b.terminal(xf, key, 2, tn[t])
        end1 = b.add("PowerTransformerEnd", f"{key}/E1", f"TWT-{f}-{t}-{i}_1", _end(
            xf, t1, 1, base_voltages[kv[f]], rated_u=tap * kv[f], rated_s=s_base, r=0.0, x=0.0))
        b.add("PowerTransformerEnd", f"{key}/E2", f"TWT-{f}-{t}-{i}_2", _end(
            xf, t2, 2, base_voltages[kv[t]], rated_u=kv[t], rated_s=s_base, r=br[BR_R] * z2, x=br[BR_X] * z2))
        if br[BR_B]:
            for side, node, b_pu in (("F", f, br[BR_B] / 2 / tap ** 2), ("T", t, br[BR_B] / 2)):
                sh = b.add("LinearShuntCompensator", f"{key}/chg{side}", f"TWT-{f}-{t}-{i}_CHG_{side}",
                           _shunt(vl[node], kv[node], g=0.0, b=b_pu * s_base / kv[node] ** 2))
                b.terminal(sh, f"{key}/chg{side}", 1, tn[node])
        if br[ANGLE] != 0:
            table = b.add("PhaseTapChangerTable", f"{key}/ptct", f"TWT-{f}-{t}-{i}_PTCT", {})
            b.add("PhaseTapChangerTablePoint", f"{key}/ptct/0", None, {
                "PhaseTapChangerTablePoint.PhaseTapChangerTable": table, "TapChangerTablePoint.step": 1,
                "TapChangerTablePoint.ratio": 1.0, "PhaseTapChangerTablePoint.angle": float(br[ANGLE]),
                "TapChangerTablePoint.r": 0.0, "TapChangerTablePoint.x": 0.0, "TapChangerTablePoint.b": 0.0,
                "TapChangerTablePoint.g": 0.0})
            b.add("PhaseTapChangerTabular", f"{key}/ptc", f"TWT-{f}-{t}-{i}_PTC", {
                "PhaseTapChanger.TransformerEnd": end1, "PhaseTapChangerTabular.PhaseTapChangerTable": table,
                "TapChanger.lowStep": 1, "TapChanger.highStep": 1, "TapChanger.neutralStep": 1,
                "TapChanger.normalStep": 1, "TapChanger.step": 1.0, "TapChanger.neutralU": tap * kv[f],
                "TapChanger.ltcFlag": False, "TapChanger.controlEnabled": False})

    for row in energized:
        n = int(row[BUS_I])
        if row[PD] or row[QD]:
            load = b.add("EnergyConsumer", f"load/{n}", f"LOAD-{n}", {
                "Equipment.EquipmentContainer": vl[n], "EnergyConsumer.p": float(row[PD]),
                "EnergyConsumer.q": float(row[QD]), "Equipment.inService": True})
            b.terminal(load, f"load/{n}", 1, tn[n])
        if row[GS] or row[BS]:
            shunt = b.add("LinearShuntCompensator", f"shunt/{n}", f"SHUNT-{n}",
                          _shunt(vl[n], kv[n], g=float(row[GS]) / kv[n] ** 2, b=float(row[BS]) / kv[n] ** 2))
            b.terminal(shunt, f"shunt/{n}", 1, tn[n])

    slack_done = False
    for i, g in enumerate(gen):
        n = int(g[GEN_BUS])
        if n not in kv:
            continue
        online = g[GEN_STATUS] > 0
        regulating = online and btype[n] in (PV, REF)
        key = f"gen/{i}"
        unit = b.add("GeneratingUnit", f"{key}/unit", f"GU-{n}-{i}", {
            "Equipment.EquipmentContainer": substations[substation_of[n]], "Equipment.inService": bool(online),
            "GeneratingUnit.maxOperatingP": 9999.0, "GeneratingUnit.minOperatingP": -9999.0})
        sm = b.add("SynchronousMachine", key, f"GEN-{n}-{i}", {
            "Equipment.EquipmentContainer": vl[n], "RotatingMachine.GeneratingUnit": unit,
            "RotatingMachine.p": -float(g[PG]) if online else 0.0, "RotatingMachine.q": -float(g[QG]) if online else 0.0,
            "SynchronousMachine.minQ": float(g[QMIN]), "SynchronousMachine.maxQ": float(g[QMAX]),
            "RotatingMachine.ratedS": s_base, "RotatingMachine.ratedU": kv[n],
            "SynchronousMachine.operatingMode": "SynchronousMachineOperatingMode.generator",
            "SynchronousMachine.type": "SynchronousMachineKind.generator",
            "RegulatingCondEq.controlEnabled": bool(regulating),
            "SynchronousMachine.referencePriority": 1 if (btype[n] == REF and online and not slack_done) else 0,
            "Equipment.inService": bool(online)})
        slack_done |= btype[n] == REF and online
        term = b.terminal(sm, key, 1, tn[n], connected=bool(online))
        rc = b.add("RegulatingControl", f"{key}/rc", f"RC-{n}-{i}", {
            "RegulatingControl.Terminal": term, "RegulatingControl.mode": "RegulatingControlModeKind.voltage",
            "RegulatingControl.discrete": False, "RegulatingControl.enabled": bool(regulating),
            "RegulatingControl.targetValue": float(g[VG]) * kv[n],
            "RegulatingControl.targetValueUnitMultiplier": "UnitMultiplier.k", "RegulatingControl.targetDeadband": 0.0})
        b.objects[sm[1:]]["RegulatingCondEq.RegulatingControl"] = rc
    return b


def _end(transformer: str, terminal: str, number: int, base_voltage: str, rated_u: float, rated_s: float,
         r: float, x: float) -> dict:
    return {"PowerTransformerEnd.PowerTransformer": transformer, "TransformerEnd.Terminal": terminal,
            "TransformerEnd.endNumber": number, "TransformerEnd.BaseVoltage": base_voltage,
            "PowerTransformerEnd.ratedU": rated_u, "PowerTransformerEnd.ratedS": rated_s,
            "PowerTransformerEnd.r": r, "PowerTransformerEnd.x": x, "PowerTransformerEnd.b": 0.0,
            "PowerTransformerEnd.g": 0.0}


def _shunt(container: str, nom_u: float, g: float, b: float) -> dict:
    return {"Equipment.EquipmentContainer": container, "ShuntCompensator.nomU": nom_u,
            "LinearShuntCompensator.gPerSection": g, "LinearShuntCompensator.bPerSection": b,
            "ShuntCompensator.maximumSections": 1, "ShuntCompensator.normalSections": 1,
            "ShuntCompensator.sections": 1.0, "RegulatingCondEq.controlEnabled": False,
            "Equipment.inService": True}


def write(builder: Builder, out_dir: Path, case_name: str) -> list[Path]:
    import cimoxide

    ds = cimoxide.CimDataset.decode_str(EMPTY)
    for obj in builder.objects.values():
        ds[obj["id"]] = obj
    models = {p: f"urn:uuid:{uuid.uuid5(NAMESPACE, f'{case_name}/model/{p}')}" for p in PROFILES}
    for profile, uri in PROFILES.items():
        header = {"_type": "FullModel", "id": models[profile], "Model.profile": uri,
                  "Model.scenarioTime": SCENARIO_TIME, "Model.created": SCENARIO_TIME, "Model.version": "1",
                  "Model.description": f"{case_name} converted from MATPOWER by grid-bench",
                  "Model.modelingAuthoritySet": "https://github.com/m-mirz/grid-bench"}
        if profile != "EQ":
            header["Model.DependentOn"] = models["EQ"]
        ds[models[profile]] = header
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for profile in ("EQ", "SSH", "TP"):
        path = out_dir / f"{case_name}_{profile}.xml"
        path.write_text(ds.to_xml_for_profile(profile))
        paths.append(path)
    return paths


def main(argv: list[str]) -> None:
    m_path, out_dir = Path(argv[0]), Path(argv[1])
    for p in write(convert(parse_m(m_path), m_path.stem), out_dir, m_path.stem):
        print(p)


if __name__ == "__main__":
    main(sys.argv[1:])
