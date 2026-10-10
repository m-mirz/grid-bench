# The gpusim2grid image: the shared base plus gpusim2grid (batched AC power
# flow on CUDA with cuDSS) and the lightsim2grid it is seeded from, both
# compiled here. Built by docker/build.sh in place of docker/tool.dockerfile.
#
# gpusim2grid has no release: it is built from a commit of its main branch,
# the newest when it was added (2026-10-04), taken on purpose at 6 days old
# like sparlectra and p3s, to add the tool without waiting. It needs
# lightsim2grid's C++ headers and CMake package (lightsim2grid_core), which
# the PyPI wheel does not ship, so lightsim2grid is built from its 1.0.0
# sdist, the release the lightsim2grid image installs from PyPI, with the
# wheel's flags (no -march=native, no -O3: `compiled_march_native` and
# `compiled_o3_optim` are False in that image).
#
# Pinned like everything else, no apt packages: NVIDIA's CUDA 12.9 devel
# image by digest (nvcc, cuSPARSE, cuBLAS; Ubuntu 22.04, whose glibc 2.35 is
# older than the base image's, so the extensions run there), cuDSS 0.8.0
# (the series ExaPF.jl's GPU image uses) and both source tarballs by SHA-256,
# the build tools by uv.lock's `build` group. CUDA 12 runs on any driver
# from R525 on.
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04@sha256:bd4e2680a261c212f1e2fea241606f71497dc67a417f73175d794ec8212b5ba8 AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.33@sha256:77280f2f771df71f90786c314fe1bbc1e023feac652969bbf139c280babf2eb7 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.13.14
WORKDIR /env
COPY tool-configs/gpusim2grid/pyproject.toml tool-configs/gpusim2grid/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --only-group build --python 3.13.14
ENV PATH="/env/.venv/bin:${PATH}"

# cuDSS from NVIDIA's redistributable archive (headers, libraries, CMake
# package), the version in its redistrib_0.8.0.json.
ADD --checksum=sha256:29716a05c64b28531309afcb7a9dd5044081a43197e72bdd0fbf29e9b4ee4707 \
    https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.8.0.10_cuda12-archive.tar.xz /src/cudss.tar.xz
RUN mkdir /opt/cudss && tar -xJf /src/cudss.tar.xz -C /opt/cudss --strip-components=1

# lightsim2grid 1.0.0 from its sdist (SuiteSparse and Eigen included), with
# KLU, installed into the build environment for gpusim2grid to compile against.
ADD --checksum=sha256:d9bebf42eb753931106fe7206e29a8077a3132b79029e71d473ccdd3b3ddbdc7 \
    https://files.pythonhosted.org/packages/16/67/d12102b4283d13d33d3ccf63eaf53d22080d0e35e8d2d63ffa5d939bb3d2/lightsim2grid-1.0.0.tar.gz /src/lightsim2grid.tar.gz
RUN mkdir /src/lightsim2grid && tar -xzf /src/lightsim2grid.tar.gz -C /src/lightsim2grid --strip-components=1 \
    && uv build --wheel --no-build-isolation --python /env/.venv/bin/python -o /wheels /src/lightsim2grid \
    && uv pip install --python /env/.venv/bin/python --no-deps /wheels/lightsim2grid-*.whl

# gpusim2grid at commit 416ae9f (main, 2026-10-04), FP64 (its default).
# - CUDA architectures: Turing to Hopper and consumer Blackwell (sm_75 ... sm_120),
#   the GPUs a cloud machine has; its CMake sets none.
# - GPUSIM2GRID_SUITESPARSE_DIR: the KLU headers its lightsim2grid bridge
#   includes, from the same sdist lightsim2grid was built from.
ADD --checksum=sha256:1dcb1e6024f8cc8eca6dce6bd396fde1303282736c87cdb897481b65c3bc462a \
    https://github.com/Grid2op/gpusim2grid/archive/416ae9fc2221965f654e26154e7c628322e172e8.tar.gz /src/gpusim2grid.tar.gz
RUN mkdir /src/gpusim2grid && tar -xzf /src/gpusim2grid.tar.gz -C /src/gpusim2grid --strip-components=1 \
    && CMAKE_PREFIX_PATH=/opt/cudss uv build --wheel --no-build-isolation --python /env/.venv/bin/python -o /wheels /src/gpusim2grid \
       "--config-setting=cmake.define.CMAKE_CUDA_ARCHITECTURES=75;80;86;89;90;120" \
       --config-setting=cmake.define.GPUSIM2GRID_SUITESPARSE_DIR=/src/lightsim2grid/SuiteSparse

# The CUDA and cuDSS libraries the extension loads (cuDSS opens some of its
# own at run time, so all of its archive's libraries), collected for the
# runtime image; the driver (libcuda) comes from the host at run time. The
# extension links the CUDA runtime statically; libcudart is copied from the
# same toolkit only for the adapter to report its version.
RUN mkdir -p /opt/cuda-rt \
    && cp -a /opt/cudss/lib/libcudss*.so* /opt/cuda-rt/ \
    && cp -L /usr/local/cuda/lib64/libcudart.so.12 /opt/cuda-rt/ \
    && python -m zipfile -e /wheels/gpusim2grid-*.whl /tmp/g2g \
    && LD_LIBRARY_PATH=/opt/cudss/lib:/usr/local/cuda/lib64 ldd /tmp/g2g/gpusim2grid/*.so /opt/cudss/lib/libcudss.so.0 \
       | awk '$3 ~ "^/usr/local/cuda" {print $3}' | sort -u | xargs -r cp -L -t /opt/cuda-rt/

FROM grid-bench/base:latest
WORKDIR /env
COPY tool-configs/gpusim2grid/pyproject.toml tool-configs/gpusim2grid/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --python 3.13.14
COPY --from=build /opt/cuda-rt /opt/cuda-rt
RUN echo /opt/cuda-rt > /etc/ld.so.conf.d/cuda-rt.conf && ldconfig
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    uv pip install --python /env/.venv/bin/python --no-deps /wheels/*.whl
ENV PATH="/env/.venv/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/gpusim2grid
WORKDIR /bench
