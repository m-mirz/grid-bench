"""gpusim2grid: AC Newton-Raphson on an NVIDIA GPU (CUDA, cuDSS), seeded
from a lightsim2grid grid. Its own image (tool-configs/gpusim2grid), built
from a commit of its main branch (no release yet), with lightsim2grid 1.0.0
compiled from source next to it; results go to `results-docker/gpu/` (the
GPU tabs), next to lightsim2grid run on the same machine.

Input: lightsim2grid's `init_from_matpower` on the prepared `.mat`, exactly
as adapters/lightsim2grid_adapter.py, with its documented loss (online
generators on PQ-typed buses regulate voltage): gpusim2grid reads its
Ybus, injections and bus types off that grid through its C++ bridge, so
both solve the same imported problem. Bus ids: gpusim2grid works in
lightsim2grid's AC-solver numbering, mapped to the grid's buses (one per
MATPOWER bus, in file order) by `id_ac_solver_to_me`, asserted.

Settings:
- `AcPfGPU((Ybus, v_flat, Sbus, slack, slack_weights, pv, pq),
  init_from_n_powerflow=False, max_iter=MAX_ITERATIONS, tol=TOLERANCE_PU)`:
  its explicit-array constructor, which runs gpusim2grid's Newton loop on
  the GPU from `v_flat` to the tolerance (‖F‖∞) or the limit (the default
  instead trusts lightsim2grid's CPU solution and only checks its
  residual). The arrays are lightsim2grid's, read by gpusim2grid's own
  `extract_grid_arrays` off the grid once its base case is solved
  (`solved_grid`, untimed): AC-solver numbering, one slack. `v_flat` is the
  flat start lightsim2grid's `ac_pf` makes of all-ones: the setpoint
  magnitude at PV and slack buses (read off that solution), 1 p.u.
  elsewhere, every angle zero. Not converged (`timings.converged`) fails.
- Not `AcPfGPU.ac_pf(v_flat, ...)`, its lightsim2grid-style entry point:
  at 416ae9f it crashes the process (segmentation fault in the bridge's
  session constructor) after re-seeding the grid with zero CPU iterations,
  on case14 already; so does building a bridge session from a grid that
  was never solved. The explicit-array path solves the bare [pvpq | pq]
  system, which with the case's one slack is the same system (no
  distributed slack, HVDC, SVC or remote control here).
- Not warm: a single solve is a new GPU session (upload, cuDSS analysis,
  factorization, Newton loop), as gpusim2grid offers no re-solve of an
  existing single session; the time includes that setup and the voltages'
  copy to the host. Its batch and N-1 sessions are reused (the other
  adapters).
- FP64 (its default, the build's precision), its default cuDSS settings
  (reordering, matching, pivoting), no step damping (off on the
  explicit-array path; lightsim2grid's own solve has none by default).
- No reactive limits or outer loop: gpusim2grid has none.
- Results (A100, 416ae9f): every case lightsim2grid solves is accepted (it
  fails the same ones: its base case diverges on case1888rte and case6495rte,
  and case2848rte carries lightsim2grid's documented loss). A new session
  per solve costs at least about 26 ms (case4_dist), 180 ms on
  case9241pegase: slower than lightsim2grid's CPU solve on every case.
- One GPU, the one docker-compose passes in (`GRID_BENCH_GPU`);
  `dependencies` records it with the driver and CUDA runtime. Memory
  (`rss_*`) is host memory; device memory is not measured.
"""
import ctypes
import os

import numpy as np

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.matpower import BUS_I, parse_m
from cases.registry import CASES, mat_path


def cuda_versions() -> dict[str, str]:
    """The device, driver and CUDA runtime the numbers were measured on, from
    the CUDA libraries the process loads (gpusim2grid reports none)."""
    cuda, cudart = ctypes.CDLL("libcuda.so.1"), ctypes.CDLL("libcudart.so.12")
    assert cuda.cuInit(0) == 0, "no usable CUDA device"
    device, name, driver, runtime = ctypes.c_int(), ctypes.create_string_buffer(256), ctypes.c_int(), ctypes.c_int()
    assert cuda.cuDeviceGet(ctypes.byref(device), 0) == 0 and cuda.cuDeviceGetName(name, 256, device) == 0
    cuda.cuDriverGetVersion(ctypes.byref(driver))
    cudart.cudaRuntimeGetVersion(ctypes.byref(runtime))
    v = lambda x: f"{x // 1000}.{x % 1000 // 10}"
    cudss = os.path.realpath("/opt/cuda-rt/libcudss.so.0").rpartition(".so.")[2]
    return {"gpu": name.value.decode(), "cuda_driver": v(driver.value), "cuda_runtime": v(runtime.value), "cudss": cudss}


def solved_grid(case: str):
    """lightsim2grid's grid of the case, its base case solved from the flat
    start on the CPU: every gpusim2grid session is built from a solved grid
    (built from one that is not, 416ae9f's bridge crashes the process)."""
    from lightsim2grid.algorithm import AlgorithmType
    from lightsim2grid.network import init_from_matpower
    grid = init_from_matpower(str(mat_path(case)))
    grid.change_algorithm(AlgorithmType.NR_KLU)
    v_flat = np.full(len(grid.get_bus_vn_kv()), grid.get_init_vm_pu(), dtype=complex)
    if grid.ac_pf(v_flat.copy(), MAX_ITERATIONS, TOLERANCE_PU).shape[0] == 0:
        raise DidNotConverge("lightsim2grid's base case did not converge")
    return grid, v_flat


def bus_ids(case: str, grid) -> list[str]:
    """MATPOWER bus numbers of the AC solver's buses, in its order."""
    number = parse_m(CASES[case]["file"])["bus"][:, BUS_I].astype(int)
    assert len(grid.get_bus_vn_kv()) == len(number), "one lightsim2grid bus per MATPOWER bus"
    return [str(number[i]) for i in grid.id_ac_solver_to_me()]


class Gpusim2gridAdapter(SolverAdapter):
    name = "gpusim2grid"
    display_name = "gpusim2grid (GPU)"
    color = "#eb6835"   # tools/palette.py GPUSIM2GRID: lightsim2grid's orange, dash-dotted
    package = "gpusim2grid"
    modules = ("gpusim2grid", "lightsim2grid", "lightsim2grid.network", "lightsim2grid.algorithm")
    language = "c++"
    families = ("matpower", "distribution")
    settings = {"solver": "NewtonRaphson", "backend": "cuda", "linear_solver": "cudss", "precision": "fp64",
                "entry_point": "AcPfGPU (explicit arrays)", "init": "flat", "tolerance_pu": TOLERANCE_PU,
                "max_iteration": MAX_ITERATIONS}

    def dependencies(self):
        return SolverAdapter.dependencies(self) | cuda_versions()   # shared by the batch and N-1 adapters

    def load(self, case):
        from gpusim2grid._ls2g_utils import extract_grid_arrays
        grid, _ = solved_grid(case)
        d = extract_grid_arrays(grid, max_iter=MAX_ITERATIONS, tol=TOLERANCE_PU)
        v_flat = np.ones(d["n_bus"], dtype=complex)
        regulated = np.concatenate([d["slack"], d["pv"]])
        v_flat[regulated] = np.abs(d["v_converged"][regulated])
        arrays = (d["Ybus"], v_flat, d["Sbus"], d["slack"], d["slack_weights"], d["pv"], d["pq"])
        return {"arrays": arrays, "ids": bus_ids(case, grid), "v": None, "iterations": None}

    def solve(self, model):
        from gpusim2grid import AcPfGPU
        gpu = AcPfGPU(model["arrays"], init_from_n_powerflow=False, max_iter=MAX_ITERATIONS, tol=TOLERANCE_PU)
        if not gpu.timings.converged:
            raise DidNotConverge(f"not converged in {MAX_ITERATIONS} iterations")
        model["v"], model["iterations"] = gpu.solve(), int(gpu.timings.nb_iter)

    def solution(self, model, case):
        v, ids = model["v"], model["ids"]
        assert len(v) == len(ids)
        return Solution(dict(zip(ids, np.abs(v))), dict(zip(ids, np.rad2deg(np.angle(v)))), model["iterations"])
