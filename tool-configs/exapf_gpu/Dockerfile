# The exapf_gpu image: the exapf image's Julia and ExaPF.jl, plus CUDA.jl and
# CUDSS.jl for ExaPF's CUDA backend. Built by docker/build.sh in place of
# docker/tool.dockerfile; building needs no GPU, running needs an NVIDIA GPU
# and the NVIDIA Container Toolkit on the host (docker-compose.yml requests
# the device).
#
# Pinned like every other image: Julia by the official image's digest,
# Python packages by uv.lock, Julia packages by julia/Manifest.toml resolved
# against a dated registry snapshot (julia/setup.jl), and the CUDA runtime by
# julia/LocalPreferences.toml (the CUDA libraries are artifacts of their JLL
# packages, fetched by content hash like any other). Nothing comes from an
# NVIDIA base image: the host's driver is mounted in at run time by the
# container toolkit, everything else is in the depot. Everything is
# installed and precompiled here: containers have no network at run time.
FROM julia:1.12.7-bookworm@sha256:709daad7eccdb0363b080df203601988cff8c6a9581162668b76804cca407888 AS julia

FROM grid-bench/base:latest
COPY --from=julia /usr/local/julia /opt/julia
WORKDIR /env
COPY tool-configs/exapf_gpu/pyproject.toml tool-configs/exapf_gpu/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --python 3.13.14

# The project's layout as in the checkout: it takes GridBenchExaPF from the
# exapf image's directory (`[sources]`, a relative path).
COPY tool-configs/exapf/julia/GridBenchExaPF /env/tool-configs/exapf/julia/GridBenchExaPF
COPY tool-configs/exapf_gpu/julia /env/tool-configs/exapf_gpu/julia
# Precompiling GridBenchExaPF loads PythonCall in plain Julia: point it at
# this image's Python and keep CondaPkg from installing one (juliacall sets
# the same at run time). install also checks that the CUDA artifacts are in.
RUN JULIA_DEPOT_PATH=/opt/julia-depot JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/env/.venv/bin/python \
    /opt/julia/bin/julia /env/tool-configs/exapf_gpu/julia/setup.jl install /env/tool-configs/exapf_gpu/julia \
    && chmod -R a+rX /opt/julia-depot

# juliacall as in the exapf image. The first depot is writable scratch (logs,
# and GPU code CUDA.jl compiles at run time); packages, precompiled caches
# and CUDA libraries come from the read-only second one. The container
# toolkit mounts the driver's compute library (libcuda) and nvidia-smi.
ENV PATH="/env/.venv/bin:/opt/julia/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/exapf_gpu \
    JULIA_DEPOT_PATH=/tmp/julia-depot:/opt/julia-depot \
    PYTHON_JULIACALL_EXE=/opt/julia/bin/julia \
    PYTHON_JULIACALL_PROJECT=/env/tool-configs/exapf_gpu/julia \
    PYTHON_JULIACALL_LIB=/opt/julia/lib/libjulia.so.1.12 \
    PYTHON_JULIACALL_THREADS=1 \
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /bench
