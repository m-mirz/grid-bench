"""gridoxide (GPU): what its adapters share, the tool's identity and its
session, and the power flow. See adapters/gridoxide_gpu_batch_adapter.py for
the batch's method and its settings.

Power flow: `GpuBatchSolver.solve_single(model, start="flat")`, a batch of one
on a session of capacity 1, on `PowerFlowModel.from_matpower`'s model and input
as adapters/gridoxide_adapter.py, flat start and tolerance as there. One system
is cuDSS's worst case, its cost roughly fixed per system: it is here to compare
with the other GPU tools, not because a device suits a single solve.

A session (`GpuBatchSolver`) is built from a model whose base case is
solved: it takes that solution as every scenario's start, gridoxide's
`start="last"`, and keeps cuDSS's symbolic analysis and the device buffers
for `CAPACITY` scenarios, so every later solve on it costs no analysis.
"""
import ctypes
import glob
import os

import numpy as np

from adapters.gridoxide_adapter import GridoxideAdapter
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.matpower import parse_m
from cases.registry import CASES, m_path

CAPACITY = 1024
# Steps on a sweep's shared first-step factors before Newton's; see
# adapters/gridoxide_gpu_batch_adapter.py.
CHORD_STEPS = 12
CHORD_BELOW = 0.9


def cuda_versions() -> dict[str, str]:
    """The device, driver, CUDA runtime and cuDSS the numbers were measured
    on, from the libraries the process loads."""
    cuda, cudart = ctypes.CDLL("libcuda.so.1"), ctypes.CDLL("libcudart.so.12")
    assert cuda.cuInit(0) == 0, "no usable CUDA device"
    device, name, driver, runtime = ctypes.c_int(), ctypes.create_string_buffer(256), ctypes.c_int(), ctypes.c_int()
    assert cuda.cuDeviceGet(ctypes.byref(device), 0) == 0 and cuda.cuDeviceGetName(name, 256, device) == 0
    cuda.cuDriverGetVersion(ctypes.byref(driver))
    cudart.cudaRuntimeGetVersion(ctypes.byref(runtime))
    v = lambda x: f"{x // 1000}.{x % 1000 // 10}"
    (cudss,) = glob.glob("/opt/cuda-rt/libcudss.so.*.*")
    return {"gpu": name.value.decode(), "cuda_driver": v(driver.value), "cuda_runtime": v(runtime.value),
            "cudss": os.path.basename(cudss).rpartition(".so.")[2]}


def session(case: str):
    """The model of `case` with its base case solved from flat start, a GPU
    session from that solution, and the MATPOWER bus number of each of the
    model's buses."""
    import gridoxide
    model = gridoxide.PowerFlowModel.from_matpower(str(m_path(case)), tol=TOLERANCE_PU, max_iter=MAX_ITERATIONS)
    numbers = model.bus_numbers()
    assert len(numbers) == len(parse_m(CASES[case]["file"])["bus"]), "one gridoxide bus per MATPOWER bus"
    model.solve(init="flat")
    gpu = gridoxide.GpuBatchSolver(model, capacity=CAPACITY, chord_steps=CHORD_STEPS, chord_below=CHORD_BELOW)
    return model, gpu, [str(b) for b in numbers]


class GridoxideGpu:
    name = "gridoxide_gpu"
    display_name = "gridoxide (GPU)"
    color = "#8b3202"   # tools/palette.py: Rust brown, base, dash-dotted (GPU); #8b3201 is gridoxide's CPU tool
    package = "gridoxide"
    language = "rust"
    modules = ("gridoxide",)
    settings = {"solver": "NewtonRaphson", "backend": "cuda", "linear_solver": "cudss", "precision": "fp64",
                "stopping": "per scenario", "tolerance_pu": TOLERANCE_PU, "max_iteration": MAX_ITERATIONS,
                "commit": os.environ.get("GRIDOXIDE_COMMIT", "unknown")}

    def dependencies(self):
        return SolverAdapter.dependencies(self) | cuda_versions()

    def step_counts(self, model):
        """What the last solve's scenarios took, untimed: `iterations` as
        gridoxide reports it, Newton steps plus one. A fixed-step tool's
        count would be its fixed steps; this shows what stopping every
        scenario at the tolerance costs against them."""
        counts = [r.iterations for r in model["results"]]
        return {"iterations_mean": sum(counts) / len(counts), "iterations_max": max(counts),
                "iterations_min": min(counts)}


class GridoxideGpuAdapter(GridoxideGpu, SolverAdapter):
    families = GridoxideAdapter.families
    settings = GridoxideGpu.settings | {"init": "flat", "reader": "from_matpower (.m)",
                                        "entry_point": "GpuBatchSolver.solve_single", "capacity": 1}

    def load(self, case):
        import gridoxide
        model = gridoxide.PowerFlowModel.from_matpower(
            str(CASES[case]["file"]), tol=TOLERANCE_PU, max_iter=MAX_ITERATIONS)
        return {"model": model, "gpu": gridoxide.GpuBatchSolver(model, capacity=1),
                "ids": [str(b) for b in model.bus_numbers()], "result": None}

    def solve(self, model):
        result = model["gpu"].solve_single(model["model"], start="flat")
        if not result.converged:
            raise DidNotConverge(f"mismatch {result.max_mismatch} after {result.iterations} iterations")
        model["result"] = result

    def solution(self, model, case):
        r, ids = model["result"], model["ids"]
        if len(r.voltage_mag) != len(ids):
            raise RuntimeError(f"gridoxide has {len(r.voltage_mag)} buses, the case {len(ids)}")
        return Solution(dict(zip(ids, r.voltage_mag)), dict(zip(ids, np.rad2deg(r.voltage_ang))), r.iterations)
