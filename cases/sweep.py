"""Operating-point sweeps for the batch cases, generated from the case with
no tool involved.

A sweep case (`<case>#sweep`) is a MATPOWER case plus `SWEEP_SIZE`
operating points on its fixed topology, the workload of a time series or a
Monte-Carlo study: every tool solves all of them in one timed call, through
its batch API where it has one (`adapters.batch_adapter`).

Scenario k scales the case's demand and dispatch:

- loads: `Pd` and `Qd` of every bus by `scale[k] * (1 + SIGMA_BUS * e)`,
  `e` standard normal per bus and scenario, the same factor for P and Q
  (constant power factor). `scale` is one period of a sinusoid between
  `SCALE_MIN` and `SCALE_MAX`, a daily load curve without the case's peak:
  heavily loaded distribution feeders (mvlv down to 0.67 p.u.) do not
  converge much above their base point.
- generators: `Pg` of every online generator by `scale[k]` (proportional
  redispatch); the slack absorbs the noise and the losses. `Qg` and every
  voltage setpoint stay as in the case.

Zero stays zero, so no scenario creates a load or generator the case does
not have, and a tool only updates the elements its importer created.

Noise comes from `numpy.random.default_rng` seeded from the case key, as in
`cases.measurements`. `cases.prep` solves every scenario with
`cases.truth.solve_pf` and checks it with the tier-1 residual before
writing it, so every scenario provably has a solution from flat start.

Written as `.npz` (`sweep_path`): `load_bus` (MATPOWER bus numbers of the
buses with a load), `pd`, `qd` (scenarios x those buses, MW and MVAr),
`gen_row` (rows of the online generators in the `.m`'s gen matrix),
`gen_bus` (their buses), `pg` (scenarios x those rows, MW), and `scale`.
"""
import hashlib

import numpy as np

from cases.matpower import BUS_I, BUS_TYPE, GEN_BUS, GEN_STATUS, ISOLATED, PD, PG, QD

SWEEP_SIZE = 100
SCALE_MIN, SCALE_MAX = 0.6, 1.0
SIGMA_BUS = 0.05


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def generate(mpc: dict, key: str) -> dict[str, np.ndarray]:
    bus, gen = mpc["bus"], mpc["gen"]
    loaded = (bus[:, BUS_TYPE] != ISOLATED) & ((bus[:, PD] != 0) | (bus[:, QD] != 0))
    online = np.flatnonzero(gen[:, GEN_STATUS] > 0)
    t = np.arange(SWEEP_SIZE) / SWEEP_SIZE
    scale = SCALE_MIN + (SCALE_MAX - SCALE_MIN) * (0.5 - 0.5 * np.cos(2 * np.pi * t))
    factor = scale[:, None] * (1 + SIGMA_BUS * np.random.default_rng(_seed(key)).standard_normal(
        (SWEEP_SIZE, int(loaded.sum()))))
    return {
        "load_bus": bus[loaded, BUS_I].astype(np.int64),
        "pd": bus[loaded, PD][None, :] * factor, "qd": bus[loaded, QD][None, :] * factor,
        "gen_row": online.astype(np.int64), "gen_bus": gen[online, GEN_BUS].astype(np.int64),
        "pg": gen[online, PG][None, :] * scale[:, None],
        "scale": scale,
    }


def scenario(mpc: dict, sweep: dict, k: int) -> dict:
    """The case with scenario k's demand and dispatch, the problem every tool
    is asked to solve for that scenario. Rows joined by bus number."""
    out = {name: (v.copy() if isinstance(v, np.ndarray) else v) for name, v in mpc.items()}
    pos = {int(b): i for i, b in enumerate(out["bus"][:, BUS_I])}
    rows = [pos[int(b)] for b in sweep["load_bus"]]
    out["bus"][rows, PD] = sweep["pd"][k]
    out["bus"][rows, QD] = sweep["qd"][k]
    out["gen"][sweep["gen_row"], PG] = sweep["pg"][k]
    return out


def write(data: dict, path) -> None:
    with open(path, "wb") as f:
        np.savez(f, **data)


def read(path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return dict(z)
