"""Measurement sets for the state-estimation cases, generated from the true
state (`cases.truth`) with no tool involved.

A state-estimation case is a MATPOWER case plus one scenario. The scenario
says what is measured and how accurately:

- `exact`: |V| at every energized bus, and P and Q injections at every
  energized bus, all without noise. Twice as many measurements as a power
  flow has equations. The weighted least-squares optimum is the true state,
  J = 0, so this is the first check of each tool's measurement model (signs,
  units, sides) before noise hides anything.
- `noisy`: |V| at the slack and at every bus with an online generator, P and
  Q injections at every energized bus, and P and Q at the from end of every
  in-service branch, each with Gaussian noise of its own sigma. That is a
  redundancy of about 2 to 3 on transmission cases, typical of SCADA
  measurement sets.

Sigmas, the same for both scenarios (they are the weights):
- |V|: `SIGMA_VM_PU`, a 0.4 % voltage transducer.
- powers: `SIGMA_REL` of the true value plus `SIGMA_FLOOR` of base power.
  The relative part is a 2 % class meter; the floor keeps a sigma on
  zero-injection buses and small distribution loads (case33bw loads 0.06 MW
  on a 10 MVA base) without swamping them.

Zero-injection buses are ordinary measurements of 0 (with noise in `noisy`),
not constraints: not every tool can express an equality constraint, and the
problem has to be the same for every tool.

Noise comes from `numpy.random.default_rng` seeded from the case key, so a
case always gets the same measurements (numpy is pinned; the file's sha256
goes into every result so a changed stream would be visible).

Written as JSON (`measurements_path`), units as in MATPOWER: |V| in p.u.,
P and Q in MW and MVAr, injections positive into the network, branch flows
positive out of the from bus. Buses by MATPOWER number, branches by their
row in the `.m`'s branch matrix, with from and to bus as well so each
adapter can assert its join (rule 5).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from cases.matpower import BUS_TYPE, F_BUS, GEN_BUS, GEN_STATUS, ISOLATED, REF, T_BUS
from cases.truth import solve_pf
from oracle.ybus import make_ybus, make_yf

SCENARIOS = ("exact", "noisy")
SIGMA_VM_PU = 0.004
SIGMA_REL = 0.02
SIGMA_FLOOR = 0.001


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def generate(mpc: dict, key: str, scenario: str) -> dict:
    """The measurement set of `key` (`<case>~<scenario>`) for the parsed
    `.m` in `mpc`."""
    ids, v = solve_pf(mpc)
    base = float(mpc["baseMVA"])
    bus_type = mpc["bus"][:, BUS_TYPE].astype(int)
    energized = bus_type != ISOLATED
    vz = np.where(energized, v, 0)
    s_bus = vz * np.conj(make_ybus(mpc)[1] @ vz) * base
    rows, f, yf = make_yf(mpc)
    s_from = vz[f] * np.conj(yf @ vz) * base

    gen = mpc["gen"]
    gen_buses = {int(b) for b in gen[gen[:, GEN_STATUS] > 0, GEN_BUS]}
    vm_at = [i for i in range(len(ids)) if energized[i]
             and (scenario == "exact" or bus_type[i] == REF or int(ids[i]) in gen_buses)]
    branch = mpc["branch"]
    pos = {int(b): i for i, b in enumerate(ids)}
    live = [k for k, r in enumerate(rows)
            if energized[pos[int(branch[r, F_BUS])]] and energized[pos[int(branch[r, T_BUS])]]]

    meas = []
    for i in vm_at:
        meas.append({"kind": "vm", "bus": int(ids[i]), "value": float(abs(v[i])), "sigma": SIGMA_VM_PU})
    for i in np.flatnonzero(energized):
        for kind, x in (("p_inj", s_bus[i].real), ("q_inj", s_bus[i].imag)):
            meas.append({"kind": kind, "bus": int(ids[i]), "value": float(x), "sigma": _sigma(x, base)})
    if scenario == "noisy":
        for k in live:
            r = int(rows[k])
            for kind, x in (("p_from", s_from[k].real), ("q_from", s_from[k].imag)):
                meas.append({"kind": kind, "branch_row": r, "from_bus": int(branch[r, F_BUS]),
                             "to_bus": int(branch[r, T_BUS]), "value": float(x), "sigma": _sigma(x, base)})
        noise = np.random.default_rng(_seed(key)).standard_normal(len(meas))
        for m, e in zip(meas, noise):
            m["value"] += float(e) * m["sigma"]

    on = np.flatnonzero(energized)
    return {
        "case": key, "scenario": scenario, "baseMVA": base,
        "slack": int(ids[bus_type == REF][0]),
        "true_state": {str(int(ids[i])): [float(abs(v[i])), float(np.rad2deg(np.angle(v[i])))] for i in on},
        "measurements": meas,
    }


def _sigma(x: float, base: float) -> float:
    return SIGMA_REL * abs(x) + SIGMA_FLOOR * base


def write(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=0))


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())
