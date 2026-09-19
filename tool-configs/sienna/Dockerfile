# The sienna image: the shared base (Python via uv) plus Julia and the Sienna
# packages. Built by docker/build.sh in place of docker/tool.dockerfile.
#
# Pinned like every other image: Julia by the official image's digest (only
# its self-contained /usr/local/julia is copied; it needs nothing but glibc),
# Python packages by uv.lock, Julia packages by julia/Manifest.toml resolved
# against a dated registry snapshot (julia/setup.jl). Everything is installed
# and precompiled here: containers have no network at run time.
FROM julia:1.12.7-bookworm@sha256:709daad7eccdb0363b080df203601988cff8c6a9581162668b76804cca407888 AS julia

FROM grid-bench/base:latest
COPY --from=julia /usr/local/julia /opt/julia
WORKDIR /env
COPY tool-configs/sienna/pyproject.toml tool-configs/sienna/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --python 3.13.14

COPY tool-configs/sienna/julia /env/julia
# Precompiling GridBenchSienna loads PythonCall in plain Julia: point it at
# this image's Python and keep CondaPkg from installing one (juliacall sets
# the same at run time).
RUN JULIA_DEPOT_PATH=/opt/julia-depot JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/env/.venv/bin/python \
    /opt/julia/bin/julia /env/julia/setup.jl install /env/julia \
    && chmod -R a+rX /opt/julia-depot

# juliacall: this Julia and this project, as installed. With EXE, PROJECT and
# LIB set, juliacall bypasses juliapkg (which would resolve the project and
# needs it writable) and loads libjulia directly. The first depot is writable
# scratch (logs); packages and precompiled caches come from the read-only
# second one.
ENV PATH="/env/.venv/bin:/opt/julia/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/sienna \
    JULIA_DEPOT_PATH=/tmp/julia-depot:/opt/julia-depot \
    PYTHON_JULIACALL_EXE=/opt/julia/bin/julia \
    PYTHON_JULIACALL_PROJECT=/env/julia \
    PYTHON_JULIACALL_LIB=/opt/julia/lib/libjulia.so.1.12 \
    PYTHON_JULIACALL_THREADS=1 \
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes
WORKDIR /bench
