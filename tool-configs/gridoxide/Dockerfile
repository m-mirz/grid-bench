# The gridoxide image: the shared base plus gridoxide compiled from source
# with its SuiteSparse KLU backend. Built by docker/build.sh in place of
# docker/tool.dockerfile.
#
# The source is a named build context, `gridoxide`: a tree of the commit
# under test, which docker/build.sh takes from GRIDOXIDE_SRC (see there).
# Its commit is recorded in the image (GRIDOXIDE_COMMIT) and with every
# result (adapters/gridoxide_adapter.py).
#
# Pinned like the other compiled tools: the Rust image by digest (Debian
# bookworm, the base image's Debian, so the extension needs no newer glibc),
# Python and maturin by uv.lock's `build` group, the Rust crates by the
# source tree's Cargo.lock (`--locked`). One apt package, in the builder
# only: libclang, which bindgen needs to generate KLU's bindings.
FROM rust:1.95-bookworm@sha256:6258907abe69656e41cd992e0b705cdcfabcbbe3db374f92ed2d47121282d4a1 AS build
RUN apt-get update && apt-get install -y --no-install-recommends libclang-dev && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.33@sha256:77280f2f771df71f90786c314fe1bbc1e023feac652969bbf139c280babf2eb7 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.13.14
WORKDIR /env
COPY tool-configs/gridoxide/pyproject.toml tool-configs/gridoxide/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --only-group build --python 3.13.14
ENV PATH="/env/.venv/bin:${PATH}"

# gridoxide's release profile (opt-level 3), with:
# - `python`: the bindings, which include the MATPOWER reader;
# - `klu`: SuiteSparse KLU, vendored in the source tree and compiled by the
#   `cc` crate at the same opt-level. lightsim2grid and p3s factor with KLU
#   too, so the columns differ by their code, not their sparse solver.
# No target-cpu=native: a portable build, like every other tool here.
COPY --from=gridoxide . /src/gridoxide
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cd /src/gridoxide && maturin build --release --locked --features python,klu \
       --interpreter /env/.venv/bin/python --out /wheels

FROM grid-bench/base:latest
ARG GRIDOXIDE_COMMIT=unknown
WORKDIR /env
COPY tool-configs/gridoxide/pyproject.toml tool-configs/gridoxide/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --python 3.13.14
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    uv pip install --python /env/.venv/bin/python --no-deps /wheels/gridoxide-*.whl \
    && /env/.venv/bin/python -c "import gridoxide; gridoxide.PowerFlowModel.from_matpower"
ENV PATH="/env/.venv/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/gridoxide \
    GRIDOXIDE_COMMIT=${GRIDOXIDE_COMMIT}
WORKDIR /bench
