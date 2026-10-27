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
# source tree's Cargo.lock (`--locked`), IPOPT and MUMPS by release tag.
# In the builder, libclang (bindgen generates the KLU and IPOPT bindings)
# and the compilers and BLAS/LAPACK IPOPT is built with; in the runtime
# image, the three shared libraries it then needs.
FROM rust:1.95-bookworm@sha256:6258907abe69656e41cd992e0b705cdcfabcbbe3db374f92ed2d47121282d4a1 AS build
RUN apt-get update && apt-get install -y --no-install-recommends libclang-dev gfortran \
        libblas-dev liblapack-dev pkg-config && rm -rf /var/lib/apt/lists/*

# IPOPT for the AC-OPF benchmark (gridoxide's `opf-ipopt` feature), built from
# source: bookworm packages IPOPT 3.11.9, whose C interface predates the
# `ipindex`/`ipnumber` types gridoxide's bindings are generated against.
# 3.14.19, the Ipopt JuliaGrid's image runs, with MUMPS as its linear solver
# (sequential, no MPI), from ThirdParty-Mumps 3.0.9.
RUN git clone --depth 1 --branch releases/3.0.9 https://github.com/coin-or-tools/ThirdParty-Mumps /tmp/mumps \
    && cd /tmp/mumps && ./get.Mumps && ./configure --prefix=/opt/ipopt && make -j"$(nproc)" && make install \
    && git clone --depth 1 --branch releases/3.14.19 https://github.com/coin-or/Ipopt /tmp/ipopt \
    && cd /tmp/ipopt && ./configure --prefix=/opt/ipopt --with-mumps --disable-java \
       PKG_CONFIG_PATH=/opt/ipopt/lib/pkgconfig && make -j"$(nproc)" && make install \
    && rm -rf /tmp/mumps /tmp/ipopt
ENV IPOPT_ROOT=/opt/ipopt LD_LIBRARY_PATH=/opt/ipopt/lib
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
# - `opf-ipopt`: the AC-OPF's IPOPT backend, linked against /opt/ipopt,
#   which the runtime image copies; so the wheel is not repaired
#   (`--auditwheel skip`) and is only for this image.
# No target-cpu=native: a portable build, like every other tool here.
COPY --from=gridoxide . /src/gridoxide
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cd /src/gridoxide && maturin build --release --locked --features python,klu,opf-ipopt \
       --auditwheel skip --interpreter /env/.venv/bin/python --out /wheels

FROM grid-bench/base:latest
ARG GRIDOXIDE_COMMIT=unknown
RUN apt-get update && apt-get install -y --no-install-recommends libgfortran5 libblas3 liblapack3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/ipopt/lib /opt/ipopt/lib
ENV LD_LIBRARY_PATH=/opt/ipopt/lib
WORKDIR /env
COPY tool-configs/gridoxide/pyproject.toml tool-configs/gridoxide/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --python 3.13.14
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    uv pip install --python /env/.venv/bin/python --no-deps /wheels/gridoxide-*.whl \
    && /env/.venv/bin/python -c "import gridoxide; gridoxide.PowerFlowModel.from_matpower; gridoxide.OptimalPowerFlowModel"
ENV PATH="/env/.venv/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/gridoxide \
    GRIDOXIDE_COMMIT=${GRIDOXIDE_COMMIT}
WORKDIR /bench
