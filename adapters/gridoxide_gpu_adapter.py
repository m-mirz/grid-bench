"""gridoxide (GPU): what the batch and N-1 adapters share, the tool's
identity and its session. gridoxide has no power-flow adapter here: its
GPU path solves batches only (a single solve would be one system on a
device made for thousands). See adapters/gridoxide_gpu_batch_adapter.py
for the method and its settings.

A session (`GpuBatchSolver`) is built from a model whose base case is
solved: it takes that solution as every scenario's start, gridoxide's
`start="last"`, and keeps cuDSS's symbolic analysis and the device buffers
for `CAPACITY` scenarios, so every later solve on it costs no analysis.
"""
import ctypes
import glob
import os

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, SolverAdapter
from cases.matpower import parse_m
from cases.registry import CASES, m_path

CAPACITY = 1024


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
    return model, gridoxide.GpuBatchSolver(model, capacity=CAPACITY), [str(b) for b in numbers]


class GridoxideGpu:
    name = "gridoxide_gpu"
    display_name = "gridoxide (GPU)"
    color = "#8b3202"   # tools/palette.py: Rust brown, base, dash-dotted (GPU); #8b3201 is gridoxide's CPU tool
    package = "gridoxide"
    language = "rust"
    modules = ("gridoxide",)
    settings = {"solver": "NewtonRaphson", "backend": "cuda", "linear_solver": "cudss", "precision": "fp64",
                "stopping": "per scenario", "tolerance_pu": TOLERANCE_PU, "max_iteration": MAX_ITERATIONS}

    def dependencies(self):
        return SolverAdapter.dependencies(self) | cuda_versions()
