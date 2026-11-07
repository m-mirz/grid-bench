"""gridoxide (GPU): the same polar AC-OPF as adapters/gridoxide_opf_adapter.py,
which this subclasses, solved by gridoxide's own interior point on the device
instead of IPOPT: `OptimalPowerFlowModel.from_matpower(solver="nlp-gpu")`.

Input, joins, flat start, `tol = OPF_TOLERANCE`, `max_iter =
OPF_MAX_ITERATIONS`: the CPU adapter's, unchanged (its docstring). Here:
- `solver="nlp-gpu"`: the model's evaluation, the interior point's vectors and
  the KKT system on one GPU, by CUDA kernels and cuDSS (`LDL^T` of the
  augmented system, its inertia read from cuDSS); the decisions on the host,
  from the scalars that come back. The restoration phase runs on the host's
  vectors. Compared with examodelspower_gpu: no relaxed equalities or bounds,
  so the balances are met as on the CPU, and IPOPT's algorithm rather than
  MadNLP's.
- `deterministic=False`: cuDSS as MadNLP's runs by default, with its faster
  solves. The iteration count may vary from run to run, so a case's rounds
  are not the same path; with `True` a solve is reproducible and takes longer.
- The model is built once in `load` (timed as import); every `solve` makes its
  own device buffers and runs cuDSS's analysis, so a solve's time includes
  them, as examodelspower_gpu's "a new solver per solve" does.
- `CUDSS_THREADING_LIB` in the image: cuDSS's analysis on every core.
"""
import os

from adapters.gridoxide_gpu_adapter import GridoxideGpu, cuda_versions
from adapters.gridoxide_opf_adapter import GridoxideOptimizer
from adapters.optimizer_adapter import OPF_MAX_ITERATIONS, OPF_TOLERANCE, OptimizerAdapter
from cases.registry import CASES


class GridoxideGpuOptimizer(GridoxideOptimizer):
    name = GridoxideGpu.name
    display_name = "gridoxide (GPU)"
    color = GridoxideGpu.color
    settings = {"formulation": "polar, from_matpower (.m)", "solver": "nlp (IPOPT's algorithm)",
                "linear_solver": "cudss", "backend": "cuda", "deterministic": False, "init": "flat",
                "tol": OPF_TOLERANCE, "max_iter": OPF_MAX_ITERATIONS,
                "commit": os.environ.get("GRIDOXIDE_COMMIT", "unknown")}

    def dependencies(self):
        return OptimizerAdapter.dependencies(self) | cuda_versions()

    def load(self, case):
        import gridoxide
        model = gridoxide.OptimalPowerFlowModel.from_matpower(
            str(CASES[case]["file"]), solver="nlp-gpu", deterministic=False,
            tol=OPF_TOLERANCE, max_iter=OPF_MAX_ITERATIONS)
        return {"model": model}
