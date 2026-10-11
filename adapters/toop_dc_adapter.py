"""ToOp (Elia Group's topology optimizer): DC power flow and DC N-1 with its
JAX DC solver on an NVIDIA GPU. Its own image (tool-configs/toop), built
from a commit of its main branch (no release, not on PyPI). DC only: ToOp
has no AC solver of its own (it validates in AC through pandapower or
PowSyBl).

ToOp computes flows from sensitivities: PTDF for the base case, LODF for
every outage, batched on the GPU; it computes no bus angles. It is graded on
its flows alone (`DcSolution.bus_ids = None`, `oracle.dc`'s flows-only
verdict), which on these connected grids determine the same solution.

Input: ToOp's PowSyBl backend, which reads a grid folder: `grid.xiidm`,
the case through PowSyBl's MATPOWER importer on the prepared `.mat` (as
adapters/pypowsybl_adapter.py, its network saved as XIIDM), masks of the
lines and transformers to monitor (all) and to outage (none for DC, the
case's outages for DC N-1), in the order the backend reads each mask
(lines in `get_lines` order, which it sorts only after assigning the
masks; transformers sorted, `sort_powsybl_element_frame_by_id`; the
outages ToOp then takes are asserted against the case's),
no relevant substation (nothing is to be split), and an empty asset
topology (the master-data file ToOp's importer would write, with no bus
group). The folder is written in a temporary directory at load; its
preprocessing (`preprocess`, `convert_to_jax`, ToOp's own steps, not
`load_grid`, which also saves its artifacts and needs package metadata the
image does not install) is part of the timed load.

Why not ToOp's pandapower backend (the first version): pandapower's
`from_mpc` makes some branch rows pandapower impedances (case118 2, case300
66, case2869pegase 26, case2848rte 219), which that backend models but can
neither monitor nor outage, and is lossy on case300 and case3120sp. It also
showed two ToOp issues, reported: station names made unique in cubic time
(an unnamed grid of 9241 buses did not load in 20 min), and `get_node_ids`
indexing pandapower's bus ids with the pp->ppc lookup instead of its inverse
(node labels permuted where pandapower reorders buses, case2848rte).

Join: ToOp's branch ids are the IIDM ids (`LINE-<f>-<t>`, `TWT-<f>-<t>`),
joined to branch rows by the N-1 adapter's `branch_ids` (asserted). ToOp's
from node is side 1 (`bus1_id`), and side 1 of every element is asserted to
be its row's from bus (`bus_breaker_bus1_id` is `BUS-<f>`), so no row is
reversed.

Settings:
- The backend's load-flow parameters: the pypowsybl adapter's (one slack,
  the case's; ToOp's default distributes the slack), for its initial AC
  and DC load flows, which feed its AC/DC comparison, not its DC model.
  `fail_on_non_convergence=False`: case2848rte's initial AC load flow does
  not converge (as pypowsybl's AC column shows), which only leaves that
  comparison empty.
- `jax_enable_x64`: double precision (ToOp's own examples set it). Measured
  in single precision (A100): case1354pegase#dcn1-full is off by 2.6e-3 MW,
  over the 1e-3 MW tolerance, at 49 against 54 ms per call;
  case9241pegase#dcn1-full takes 478 against 891 ms. A GeForce card runs
  double precision at 1/64 of single (the A100 at 1/2).
- `batch_size_bsdf = 1`: one topology, the unsplit grid
  (`default_topology`), as ToOp's own `run_initial_loadflow` does;
  its default of 16 would solve 16 copies.
- The solve: `compute_symmetric_batch` with no disconnections and no
  injection changes, which returns the base flows (`n_0_matrix`) and every
  outage's (`n_1_matrix`, simple branch outages first, in branch order) in
  one call; the result copied to the host is part of the timed call. The
  first call compiles (JIT), which the benchmark's warm-up absorbs.
- Sign: ToOp's flow on a branch is minus PowSyBl's side-1 flow (its
  backend's own convention, `get_basecase_dc_branch_flows`); reported as
  the from-end flow of the row.
- GPU: JAX's default device; `dependencies` records it (the site's GPU
  detection), and JAX's and CUDA's versions.

Findings (A100):
- ToOp's DC model takes its injections from generators and loads, not from
  shunt compensators, so shunt conductance (Gs, which MATPOWER counts as
  load) is left out, as in OpenLoadFlow's DC (adapters/pypowsybl_dc_adapter.py):
  case300 differs by 1.3 MW, case2869pegase by 2.45 MW. Its check that no
  shunt carries active power does not fire on them.
- Exact (to 1e-10 MW, every branch reported) on case14, case118,
  case1354pegase, case3120sp and case2848rte (its phase shifters included).
- Time: about 49 ms per call on every case, case14 to case9241pegase, the
  base case alone or with 200 outages: a fixed cost of the call (JAX's
  dispatch of the jitted batch and the copy to the host), not the solve.
"""
import tempfile
from pathlib import Path

import numpy as np

from adapters.batch_adapter import base_case, outages
from adapters.dc_adapter import DcAdapter, DcContingencyAdapter, DcSolution
from adapters.pypowsybl_adapter import PypowsyblAdapter
from adapters.pypowsybl_n1_adapter import branch_ids
from adapters.solver_adapter import DidNotConverge
from cases.matpower import F_BUS, T_BUS, parse_m
from cases.registry import CASES

COMMIT = "2190b1b1393f6730d917e223543bf12ee48f87b4"


def _folder(net, ids: list[str], rows) -> Path:
    """ToOp's grid folder for the network, outaging the branches of `rows`."""
    from toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers import get_lines, get_network_as_pu, get_trafos
    from toop_engine_grid_helpers.asset_topology_helpers import save_master_asset_topology
    from toop_engine_grid_helpers.powsybl.powsybl_helpers import sort_powsybl_element_frame_by_id
    from toop_engine_interfaces.asset_topology.asset_topology import MasterAssetTopology
    from toop_engine_interfaces.folder_structure import NETWORK_MASK_NAMES, PREPROCESSING_PATHS
    d = Path(tempfile.mkdtemp(prefix="gb-toop-"))
    net.save(str(d / PREPROCESSING_PATHS["grid_file_path_powsybl"]), format="XIIDM")
    masks = d / PREPROCESSING_PATHS["masks_path"]
    masks.mkdir()
    pu = get_network_as_pu(net)
    out = {ids[r] for r in rows}
    # the backend assigns line masks before sorting its lines, transformer masks after sorting them
    for kind, frame in (("line", get_lines(net, pu)),
                        ("trafo", sort_powsybl_element_frame_by_id(get_trafos(net, pu)))):
        np.save(masks / NETWORK_MASK_NAMES[f"{kind}_for_reward"], np.ones(len(frame), bool))
        np.save(masks / NETWORK_MASK_NAMES[f"{kind}_for_nminus1"], frame.index.isin(out))
    save_master_asset_topology(filename=d / PREPROCESSING_PATHS["asset_topology_master_data_file_path"],
                               master_data=MasterAssetTopology(topology_id="grid-bench", bus_groups=[]))
    return d


def _load(case: str, rows) -> dict:
    """ToOp's static information for the case, and per branch row its
    monitored column, per outage its row of the N-1 matrix."""
    import jax
    from dataclasses import replace
    from fsspec.implementations.dirfs import DirFileSystem
    from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax
    from toop_engine_dc_solver.preprocess.powsybl.powsybl_backend import PowsyblBackend
    from toop_engine_dc_solver.preprocess.preprocess import preprocess
    from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
    jax.config.update("jax_enable_x64", True)
    mpc = parse_m(CASES[case]["file"])
    branch = mpc["branch"]
    model = PypowsyblAdapter().load(case)
    net = model["network"]
    ids = branch_ids(net, branch)
    d = _folder(net, ids, rows)
    side1 = net.get_branches(attributes=["bus_breaker_bus1_id", "bus_breaker_bus2_id"])
    for r, i in enumerate(ids):
        assert side1.at[i, "bus_breaker_bus1_id"] == f"BUS-{int(branch[r, F_BUS])}" and \
            side1.at[i, "bus_breaker_bus2_id"] == f"BUS-{int(branch[r, T_BUS])}", f"branch row {r}: side 1 is its from bus"
    backend = PowsyblBackend(DirFileSystem(str(d)), lf_params=model["params"], fail_on_non_convergence=False)
    nd = preprocess(backend, parameters=PreprocessParameters())
    si = convert_to_jax(network_data=nd)
    si = replace(si, solver_config=replace(si.solver_config, batch_size_bsdf=1))
    di = si.dynamic_information
    branch_index = {b: k for k, b in enumerate(nd.branch_ids)}
    monitored = {int(b): j for j, b in enumerate(np.asarray(di.branches_monitored))}
    col = np.array([monitored[branch_index[i]] for i in ids])   # raises if ToOp left a branch out
    failed = {int(b): j for j, b in enumerate(np.asarray(di.branches_to_fail))}
    assert sorted(nd.branch_ids[k] for k in failed) == sorted(ids[r] for r in rows), "ToOp outages the case's branches"
    order = np.array([failed[branch_index[ids[r]]] for r in rows], dtype=int)
    return {"si": si, "col": col, "order": order, "n0": None, "n1": None}


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


def _from_end(flows: np.ndarray, col: np.ndarray) -> np.ndarray:
    """ToOp's flows (... x monitored) as from-end flows per branch row."""
    return -flows[..., col]


class _Toop:
    name = "toop"
    display_name = "ToOp (JAX, GPU)"
    color = "#2a78d5"   # tools/palette.py TOOP: pandapower's blue, dash-dotted (GPU)
    package = "toop_engine_dc_solver"
    language = "python"
    modules = ("jax", "pypowsybl", "pypowsybl.network", "toop_engine_dc_solver.jax.compute_batch",
               "toop_engine_dc_solver.preprocess.preprocess", "toop_engine_dc_solver.preprocess.convert_to_jax",
               "toop_engine_dc_solver.preprocess.powsybl.powsybl_backend")

    def version(self):
        return COMMIT[:7]

    def dependencies(self):
        import jax
        import jaxlib
        device = jax.devices()[0]
        return {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "pypowsybl": __import__("pypowsybl").__version__,
                "gpu": device.device_kind if device.platform == "gpu" else None, "commit": COMMIT}


class ToopDc(_Toop, DcAdapter):
    settings = {"solver": "compute_symmetric_batch (PTDF)", "jax_enable_x64": True, "batch_size_bsdf": 1,
                "angles": "not computed (graded on flows)", "backend": "PowSyBl (MATPOWER importer)"}

    def load(self, case):
        return _load(base_case(case), [])

    def solve(self, model):
        _solve(model)

    def solution(self, model, case):
        pf = _from_end(model["n0"], model["col"])
        return DcSolution(None, None, list(range(len(pf))), pf)


class ToopDcN1(_Toop, DcContingencyAdapter):
    mode = "native"
    threaded = False
    settings = {"solver": "compute_symmetric_batch (PTDF, LODF)", "jax_enable_x64": True, "batch_size_bsdf": 1,
                "angles": "not computed (graded on flows)", "backend": "PowSyBl (MATPOWER importer)", "device": "one GPU"}

    def load(self, case):
        return _load(base_case(case), [int(r) for r in outages(case)["branch_row"]])

    def solve(self, model, threads=1):
        assert threads == 1, "one GPU, no thread setting"
        _solve(model)

    def solution(self, model, case):
        pf = _from_end(model["n1"][model["order"]], model["col"])
        return DcSolution(None, None, list(range(pf.shape[1])), pf)
