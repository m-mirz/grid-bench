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

Requires cimoxide >= 0.3.2 (pinned: 0.3.3): earlier versions wrote TopologicalNodes in TP as
bare references, dropped `Equipment.inService` for lines and transformers,
and synthesized headers PowSyBl ignores. This module patched those until the
fixes were released.
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


class Builder:
    """Accumulates CIM objects keyed by deterministic UUIDs (uuid5 of a
    readable key), so the same case always converts to the same files."""

    def __init__(self, case_name: str):
        self.case = case_name
        self.objects: dict[str, dict] = {}

    def add(self, cls: str, key: str, name: str, **fields) -> str:
        m = str(uuid.uuid5(NAMESPACE, f"{self.case}/{cls}/{key}"))
        # cimoxide's structs hold these IdentifiedObject strings as required
        # fields, although CGMES treats them as optional.
        self.objects[m] = {"_type": cls, "id": f"_{m}", "m_rid": m, "name": name, "description": "",
                           "short_name": "", "energy_ident_code_eic": "", **fields}
        return f"_{m}"

    def terminal(self, equipment: str, key: str, seq: int, node: str, connected: bool = True) -> str:
        return self.add("Terminal", f"{key}/T{seq}", f"{key} T{seq}", conducting_equipment=equipment,
                        sequence_number=seq, topological_node=node, connected=connected)


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

    region = b.add("GeographicalRegion", "region", case_name)
    subregion = b.add("SubGeographicalRegion", "subregion", case_name, region=region)
    base_voltages = {v: b.add("BaseVoltage", f"bv/{v:g}", f"{v:g} kV", nominal_voltage=v) for v in sorted(set(kv.values()))}
    substation_of = _substations(mpc, energized[:, BUS_I])
    substations = {s: b.add("Substation", f"sub/{s}", f"SUB-{s}", region=subregion) for s in set(substation_of.values())}
    vl, tn = {}, {}
    for n in kv:
        vl[n] = b.add("VoltageLevel", f"vl/{n}", f"VL-{n}", base_voltage=base_voltages[kv[n]],
                      substation=substations[substation_of[n]])
        tn[n] = b.add("TopologicalNode", f"tn/{n}", f"BUS-{n}", base_voltage=base_voltages[kv[n]],
                      connectivity_node_container=vl[n])

    for i, br in enumerate(mpc["branch"]):
        if br[BR_STATUS] == 0:
            continue
        f, t = int(br[F_BUS]), int(br[T_BUS])
        key = f"br/{i}"
        if not _is_transformer(br, kv[f], kv[t]):
            z_base = kv[f] ** 2 / s_base
            line = b.add("ACLineSegment", key, f"LINE-{f}-{t}-{i}", base_voltage=base_voltages[kv[f]],
                         r=br[BR_R] * z_base, x=br[BR_X] * z_base, bch=br[BR_B] / z_base, gch=0.0,
                         length=1.0, in_service=True)
            b.terminal(line, key, 1, tn[f])
            b.terminal(line, key, 2, tn[t])
            continue
        tap = br[RATIO] if br[RATIO] != 0 else 1.0
        z2 = kv[t] ** 2 / s_base
        xf = b.add("PowerTransformer", key, f"TWT-{f}-{t}-{i}", equipment_container=substations[substation_of[f]],
                   in_service=True)
        t1 = b.terminal(xf, key, 1, tn[f])
        t2 = b.terminal(xf, key, 2, tn[t])
        end1 = b.add("PowerTransformerEnd", f"{key}/E1", f"TWT-{f}-{t}-{i}_1", power_transformer=xf, terminal=t1,
                     end_number=1, base_voltage=base_voltages[kv[f]], rated_u=tap * kv[f], rated_s=s_base,
                     r=0.0, x=0.0, b=0.0, g=0.0)
        b.add("PowerTransformerEnd", f"{key}/E2", f"TWT-{f}-{t}-{i}_2", power_transformer=xf, terminal=t2,
              end_number=2, base_voltage=base_voltages[kv[t]], rated_u=kv[t], rated_s=s_base,
              r=br[BR_R] * z2, x=br[BR_X] * z2, b=0.0, g=0.0)
        if br[BR_B]:
            for side, node, b_pu in (("F", f, br[BR_B] / 2 / tap ** 2), ("T", t, br[BR_B] / 2)):
                sh = b.add("LinearShuntCompensator", f"{key}/chg{side}", f"TWT-{f}-{t}-{i}_CHG_{side}",
                           equipment_container=vl[node], nom_u=kv[node], g_per_section=0.0,
                           b_per_section=b_pu * s_base / kv[node] ** 2, maximum_sections=1, normal_sections=1,
                           sections=1.0, control_enabled=False, in_service=True)
                b.terminal(sh, f"{key}/chg{side}", 1, tn[node])
        if br[ANGLE] != 0:
            table = b.add("PhaseTapChangerTable", f"{key}/ptct", f"TWT-{f}-{t}-{i}_PTCT")
            b.add("PhaseTapChangerTablePoint", f"{key}/ptct/0", f"TWT-{f}-{t}-{i}_PTCT_0",
                  phase_tap_changer_table=table, step=1, ratio=1.0, angle=float(br[ANGLE]), r=0.0, x=0.0, b=0.0, g=0.0)
            b.add("PhaseTapChangerTabular", f"{key}/ptc", f"TWT-{f}-{t}-{i}_PTC", transformer_end=end1,
                  phase_tap_changer_table=table, low_step=1, high_step=1, neutral_step=1, normal_step=1, step=1.0,
                  neutral_u=tap * kv[f], ltc_flag=False, control_enabled=False)

    for row in energized:
        n = int(row[BUS_I])
        if row[PD] or row[QD]:
            load = b.add("EnergyConsumer", f"load/{n}", f"LOAD-{n}", equipment_container=vl[n],
                         p=float(row[PD]), q=float(row[QD]), in_service=True)
            b.terminal(load, f"load/{n}", 1, tn[n])
        if row[GS] or row[BS]:
            shunt = b.add("LinearShuntCompensator", f"shunt/{n}", f"SHUNT-{n}", equipment_container=vl[n],
                          nom_u=kv[n], g_per_section=float(row[GS]) / kv[n] ** 2,
                          b_per_section=float(row[BS]) / kv[n] ** 2, maximum_sections=1, normal_sections=1,
                          sections=1.0, control_enabled=False, in_service=True)
            b.terminal(shunt, f"shunt/{n}", 1, tn[n])

    slack_done = False
    for i, g in enumerate(gen):
        n = int(g[GEN_BUS])
        if n not in kv:
            continue
        online = g[GEN_STATUS] > 0
        regulating = online and btype[n] in (PV, REF)
        key = f"gen/{i}"
        unit = b.add("GeneratingUnit", f"{key}/unit", f"GU-{n}-{i}", equipment_container=substations[substation_of[n]],
                     in_service=bool(online), max_operating_p=9999.0, min_operating_p=-9999.0)
        sm = b.add("SynchronousMachine", key, f"GEN-{n}-{i}", equipment_container=vl[n], generating_unit=unit,
                   p=-float(g[PG]) if online else 0.0, q=-float(g[QG]) if online else 0.0,
                   min_q=float(g[QMIN]), max_q=float(g[QMAX]), rated_s=s_base, rated_u=kv[n],
                   operating_mode="SynchronousMachineOperatingMode.generator",
                   type_="SynchronousMachineKind.generator", control_enabled=bool(regulating),
                   reference_priority=1 if (btype[n] == REF and online and not slack_done) else 0,
                   in_service=bool(online))
        slack_done |= btype[n] == REF and online
        term = b.terminal(sm, key, 1, tn[n], connected=bool(online))
        rc = b.add("RegulatingControl", f"{key}/rc", f"RC-{n}-{i}", terminal=term,
                   mode="RegulatingControlModeKind.voltage", discrete=False, enabled=bool(regulating),
                   target_value=float(g[VG]) * kv[n], target_value_unit_multiplier="UnitMultiplier.k",
                   target_deadband=0.0)
        b.objects[sm[1:]]["regulating_control"] = rc
    return b


def write(builder: Builder, out_dir: Path, case_name: str) -> list[Path]:
    import cimoxide

    ds = cimoxide.CimDataset.decode_str(EMPTY)
    for m, obj in builder.objects.items():
        ds[obj["id"]] = obj
    models = {p: f"urn:uuid:{uuid.uuid5(NAMESPACE, f'{case_name}/model/{p}')}" for p in PROFILES}
    for profile, uri in PROFILES.items():
        ds[models[profile]] = {
            "_type": "FullModel", "id": models[profile], "profile": [uri], "scenario_time": SCENARIO_TIME,
            "created": SCENARIO_TIME, "version": 1, "description": f"{case_name} converted from MATPOWER by grid-bench",
            "modeling_authority_set": "https://github.com/m-mirz/grid-bench", "supersedes": [],
            "dependent_on": [] if profile == "EQ" else [models["EQ"]],
        }
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
