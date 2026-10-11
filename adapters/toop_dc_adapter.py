"""ToOp (Elia Group's topology optimizer): DC power flow and DC N-1 with its
JAX DC solver on an NVIDIA GPU. Its own image (tool-configs/toop), built
from a commit of its main branch (no release, not on PyPI). DC only: ToOp
has no AC solver of its own (it validates in AC through pandapower or
PowSyBl).

ToOp computes flows from sensitivities: PTDF for the base case, LODF for
every outage, batched on the GPU; it computes no bus angles. It is graded on
its flows alone (`DcSolution.bus_ids = None`, `oracle.dc`'s flows-only
verdict), which on these connected grids determine the same solution.

Input: ToOp's pandapower backend, which reads a grid folder: `grid.json`,
the case through pandapower's `from_mpc` on the prepared `.mat` (as
adapters/pandapower_adapter.py), masks of the lines and transformers to
monitor (all) and to outage (none for DC, the case's outages for DC N-1),
no relevant substation (nothing is to be split), an empty asset topology
(the files ToOp's importer would write, with no bus group). The backend
builds its DC model from pandapower's ppc (`calc_b_from_branch`:
b = 1/(x·tap), phase shifts as injections). The folder is written in a
temporary directory at load; its preprocessing (`preprocess`,
`convert_to_jax`, ToOp's own steps, not `load_grid`, which also saves its
artifacts and needs package metadata the image does not install) is part of
the timed load.

Join: ToOp's branch ids are `<index>%%line` and `<index>%%trafo`,
pandapower's elements, which `from_mpc` records per branch row
(`net._from_ppc_lookups["branch"]`); pandapower's bus index is the
MATPOWER number minus 1 (`from_mpc` renumbers 0-based, asserted). Each
branch's from node in ToOp is asserted to be one of the row's ends; a
transformer runs high to low voltage in pandapower, so a row whose from bus
is ToOp's to node takes the opposite sign. The ends are read from the
backend (`get_from_nodes`, `get_to_nodes`, `get_node_ids`) by branch id,
before preprocessing merges nodes it need not keep (`REDUCED_NODE`).

Settings:
- `jax_enable_x64`: double precision (ToOp's own examples set it; in single
  precision flows are good to about 1e-4 relative, not to 1e-3 MW).
- `batch_size_bsdf = 1`: one topology, the unsplit grid
  (`default_topology`), as ToOp's own `run_initial_loadflow` does;
  its default of 16 would solve 16 copies.
- The solve: `compute_symmetric_batch` with no disconnections and no
  injection changes, which returns the base flows (`n_0_matrix`) and every
  outage's (`n_1_matrix`, simple branch outages first, in branch order) in
  one call; the result copied to the host is part of the timed call. The
  first call compiles (JIT), which the benchmark's warm-up absorbs.
- Sign: ToOp's injections are in load convention (pandapower's bus balance,
  consumption positive), so its flow on a branch is minus the from-end flow
  along its from node; reported as the from-end flow of the row.
- GPU: JAX's default device; `dependencies` records it (the site's GPU
  detection), and JAX's and CUDA's versions.
"""
import tempfile
from pathlib import Path

import numpy as np

from adapters.batch_adapter import base_case, outages
from adapters.dc_adapter import DcAdapter, DcContingencyAdapter, DcSolution
from adapters.solver_adapter import DidNotConverge
from cases.matpower import BUS_I, F_BUS, T_BUS, parse_m
from cases.registry import CASES, mat_path

COMMIT = "2190b1b1393f6730d917e223543bf12ee48f87b4"


def _folder(case: str, rows) -> tuple[Path, object]:
    """ToOp's grid folder for the case, outaging `rows` (branch rows)."""
    import pandapower as pp
    from pandapower.converter.matpower.from_mpc import from_mpc
    from toop_engine_grid_helpers.asset_topology_helpers import (save_asset_topology_bus_groups,
                                                                 save_master_asset_topology)
    from toop_engine_interfaces.asset_topology.asset_topology import MasterAssetTopology
    from toop_engine_interfaces.asset_topology.runtime_topology import RuntimeAssetTopology
    from toop_engine_interfaces.folder_structure import NETWORK_MASK_NAMES, PREPROCESSING_PATHS
    net = from_mpc(str(mat_path(case)), f_hz=50)
    lookup = net._from_ppc_lookups["branch"]
    d = Path(tempfile.mkdtemp(prefix="gb-toop-"))
    pp.to_json(net, str(d / PREPROCESSING_PATHS["grid_file_path_pandapower"]))
    masks = d / PREPROCESSING_PATHS["masks_path"]
    masks.mkdir()
    np.save(masks / NETWORK_MASK_NAMES["relevant_subs"], np.zeros(len(net.bus), bool))
    for kind in ("line", "trafo"):
        out = np.zeros(len(net[kind]), bool)
        for r in rows:
            if lookup.at[r, "element_type"] == kind:
                out[int(lookup.at[r, "element"])] = True
        np.save(masks / NETWORK_MASK_NAMES[f"{kind}_for_nminus1"], out)
        np.save(masks / NETWORK_MASK_NAMES[f"{kind}_for_reward"], np.ones(len(net[kind]), bool))
    runtime = d / PREPROCESSING_PATHS["asset_topology_runtime_file_path"]
    runtime.parent.mkdir(parents=True, exist_ok=True)
    save_asset_topology_bus_groups(filename=runtime, bus_groups=RuntimeAssetTopology(bus_groups=[]))
    save_master_asset_topology(filename=d / PREPROCESSING_PATHS["asset_topology_master_data_file_path"],
                               master_data=MasterAssetTopology(topology_id="grid-bench", bus_groups=[]))
    return d, net


def _load(case: str, rows) -> dict:
    """ToOp's static information for the case, and per branch row its
    monitored column and orientation (+1: ToOp's from node is the row's from
    bus), per outage its row of the N-1 matrix."""
    import jax
    from dataclasses import replace
    from fsspec.implementations.dirfs import DirFileSystem
    from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax
    from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
    from toop_engine_dc_solver.preprocess.preprocess import preprocess
    from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
    jax.config.update("jax_enable_x64", True)
    mpc = parse_m(CASES[case]["file"])
    number = mpc["bus"][:, BUS_I].astype(int)
    d, net = _folder(case, rows)
    assert (net.bus.index.to_numpy() == number - 1).all(), "from_mpc numbers buses 0-based, in file order"
    backend = PandaPowerBackend(DirFileSystem(str(d)))
    nd = preprocess(backend, parameters=PreprocessParameters())
    si = convert_to_jax(network_data=nd)
    si = replace(si, solver_config=replace(si.solver_config, batch_size_bsdf=1))
    di = si.dynamic_information
    lookup = net._from_ppc_lookups["branch"]
    branch_index = {b: k for k, b in enumerate(nd.branch_ids)}
    # orientation from the backend, before preprocessing merges nodes: ppc from and to bus per branch id
    node_bus = [int(n.split("%%")[0]) + 1 for n in backend.get_node_ids()]   # pandapower bus index + 1: MATPOWER number
    ends = {b: (node_bus[f], node_bus[t]) for b, f, t in
            zip(backend.get_branch_ids(), backend.get_from_nodes(), backend.get_to_nodes(), strict=True)}
    monitored = {int(b): j for j, b in enumerate(np.asarray(di.branches_monitored))}
    n = len(mpc["branch"])
    col, sign = np.full(n, -1), np.zeros(n)
    for r in range(n):
        k = branch_index.get(f"{int(lookup.at[r, 'element'])}%%{lookup.at[r, 'element_type']}")
        if k is None:
            continue   # out of service: not in ToOp's model
        f, t = ends[nd.branch_ids[k]]
        assert {f, t} == {int(mpc["branch"][r, F_BUS]), int(mpc["branch"][r, T_BUS])}, f"branch row {r}"
        col[r], sign[r] = monitored[k], 1.0 if f == int(mpc["branch"][r, F_BUS]) else -1.0
    failed = [int(b) for b in np.asarray(di.branches_to_fail)]
    rows_of = {k: i for i, k in enumerate(failed)}
    order = []
    for r in rows:
        k = branch_index[f"{int(lookup.at[r, 'element'])}%%{lookup.at[r, 'element_type']}"]
        order.append(rows_of[k])
    assert len(failed) == len(rows), f"ToOp outages {len(failed)} branches, the case {len(rows)}"
    return {"si": si, "col": col, "sign": sign, "order": np.array(order, dtype=int), "n0": None, "n1": None}


def _solve(model) -> None:
    from toop_engine_dc_solver.jax.compute_batch import compute_symmetric_batch
    from toop_engine_dc_solver.jax.topology_computations import default_topology
    si = model["si"]
    res, success = compute_symmetric_batch(topology_batch=default_topology(si.solver_config), disconnection_batch=None,
                                           injections=None, nodal_inj_start_options=None,
                                           dynamic_information=si.dynamic_information, solver_config=si.solver_config)
    if not bool(np.all(np.asarray(success))):
        raise DidNotConverge("compute_symmetric_batch reports no success")
    model["n0"], model["n1"] = np.asarray(res.n_0_matrix)[0, 0], np.asarray(res.n_1_matrix)[0, 0]


def _from_end(flows: np.ndarray, col: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """ToOp's flows (... x monitored) as from-end flows per branch row (NaN out of service)."""
    return np.where(col >= 0, -sign * flows[..., np.maximum(col, 0)], np.nan)


class _Toop:
    name = "toop"
    display_name = "ToOp (JAX, GPU)"
    color = "#2a78d5"   # tools/palette.py TOOP: pandapower's blue, dash-dotted (GPU)
    package = "toop_engine_dc_solver"
    language = "python"
    modules = ("jax", "pandapower", "pandapower.converter.matpower.from_mpc", "toop_engine_dc_solver.jax.compute_batch",
               "toop_engine_dc_solver.preprocess.preprocess", "toop_engine_dc_solver.preprocess.convert_to_jax",
               "toop_engine_dc_solver.preprocess.pandapower.pandapower_backend")

    def version(self):
        return COMMIT[:7]

    def dependencies(self):
        import jax
        import jaxlib
        device = jax.devices()[0]
        return {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "pandapower": __import__("pandapower").__version__,
                "gpu": device.device_kind if device.platform == "gpu" else None, "commit": COMMIT}


class ToopDc(_Toop, DcAdapter):
    settings = {"solver": "compute_symmetric_batch (PTDF)", "jax_enable_x64": True, "batch_size_bsdf": 1,
                "angles": "not computed (graded on flows)", "backend": "pandapower (from_mpc)"}

    def load(self, case):
        return _load(base_case(case), [])

    def solve(self, model):
        _solve(model)

    def solution(self, model, case):
        pf = _from_end(model["n0"], model["col"], model["sign"])
        return DcSolution(None, None, list(range(len(pf))), pf)


class ToopDcN1(_Toop, DcContingencyAdapter):
    mode = "native"
    threaded = False
    settings = {"solver": "compute_symmetric_batch (PTDF, LODF)", "jax_enable_x64": True, "batch_size_bsdf": 1,
                "angles": "not computed (graded on flows)", "backend": "pandapower (from_mpc)", "device": "one GPU"}

    def load(self, case):
        return _load(base_case(case), [int(r) for r in outages(case)["branch_row"]])

    def solve(self, model, threads=1):
        assert threads == 1, "one GPU, no thread setting"
        _solve(model)

    def solution(self, model, case):
        pf = _from_end(model["n1"][model["order"]], model["col"], model["sign"])
        return DcSolution(None, None, list(range(pf.shape[1])), pf)
