# The gridoxide_gpu image: the shared base plus gridoxide's Python extension
# built with its `gpu` feature (gridoxide-gpu: CUDA kernels and cuDSS's
# uniform batch). Built by docker/build.sh in place of docker/tool.dockerfile.
#
# gridoxide is built from its v0.3.0 release, the first with the GPU path:
# GitHub's tarball of the tag by SHA-256, the same source as the gridoxide
# image's (docker/build.sh). The published wheel has no `gpu` feature.
#
# Pinned like everything else, no apt packages: NVIDIA's CUDA 12.9 devel image
# by digest (nvcc; Ubuntu 22.04, whose glibc 2.35 is older than the base
# image's, so the extension runs there; gpusim2grid's image), the Rust
# toolchain from the official image by digest, cuDSS 1.0.0 (gridoxide-gpu
# refuses anything older: 0.8.0 computes NaN solutions in a uniform batch) by
# SHA-256, the crates by gridoxide's Cargo.lock (`--locked`), maturin and
# libclang (for bindgen) by uv.lock's `build` group.
FROM rust:1.98.0-slim-bookworm@sha256:1469a27c125cb5a3aebfa4f4e4665d935b02fb72cc093b2c974b3d740e43f157 AS rust

FROM nvidia/cuda:12.9.1-devel-ubuntu22.04@sha256:bd4e2680a261c212f1e2fea241606f71497dc67a417f73175d794ec8212b5ba8 AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.33@sha256:77280f2f771df71f90786c314fe1bbc1e023feac652969bbf139c280babf2eb7 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.13.14
WORKDIR /env
COPY tool-configs/gridoxide_gpu/pyproject.toml tool-configs/gridoxide_gpu/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --only-group build --python 3.13.14
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup \
    PATH="/env/.venv/bin:/usr/local/cargo/bin:${PATH}"

# cuDSS 1.0.0 from NVIDIA's redistributable archive (headers and libraries).
ADD --checksum=sha256:b8c882c3ebc8b319886f5a06e74311c5eea09846925ba5a19892de4921201746 \
    https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-1.0.0.35_cuda12-archive.tar.xz /src/cudss.tar.xz
RUN mkdir /opt/cudss && tar -xJf /src/cudss.tar.xz -C /opt/cudss --strip-components=1

# gridoxide v0.3.0 (commit e85f808). Its GPU code is that of gpu-batch's
# d105bf4 (the AC batch, on which results-docker/gpu/gridoxide_gpu-batch.json
# and -n1.json were measured) and 5d14426 (GpuDcSolver, DC N-1), squash-merged
# as gridoxide PR #29.
ADD --checksum=sha256:8e7456120a0929b4fe36c31df770d087b4acfd6a6f00e570d8cb19eeca18c804 \
    https://github.com/m-mirz/gridoxide/archive/refs/tags/v0.3.0.tar.gz /src/gridoxide.tar.gz
RUN mkdir /src/gridoxide && tar -xzf /src/gridoxide.tar.gz -C /src/gridoxide --strip-components=1

# The extension, `--features python,gpu`:
# - CUDA_ARCH=sm_80: the A100 the GPU tabs run on; gridoxide-gpu compiles its
#   kernels for one architecture, so another GPU needs this build argument
#   (sm_90 for an H100) or fails at run time with "no kernel image".
# - LIBCLANG_PATH: the libclang wheel's library. It ships no compiler
#   resource headers, so bindgen gets gcc's (stddef.h and the like) instead.
# - `--auditwheel skip`: the wheel must not bundle cuDSS or the CUDA runtime,
#   which come from /opt/cuda-rt below.
ARG CUDA_ARCH=sm_80
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cd /src/gridoxide \
    && export CUDSS_DIR=/opt/cudss CUDA_HOME=/usr/local/cuda CUDA_ARCH=${CUDA_ARCH} \
       LIBCLANG_PATH="$(python -c 'import clang, os; print(os.path.join(os.path.dirname(clang.__file__), "native"))')" \
       BINDGEN_EXTRA_CLANG_ARGS="-I$(dirname "$(gcc -print-libgcc-file-name)")/include" \
    && maturin build --release --locked --features python,gpu -i /env/.venv/bin/python \
       --compatibility linux --auditwheel skip -o /wheels

# The CUDA and cuDSS libraries the extension loads (cuDSS opens some of its
# own at run time, so all of its archive's libraries), collected for the
# runtime image; the driver (libcuda) comes from the host at run time.
RUN mkdir -p /opt/cuda-rt \
    && cp -a /opt/cudss/lib/libcudss*.so* /opt/cuda-rt/ \
    && cp -L /usr/local/cuda/lib64/libcudart.so.12 /opt/cuda-rt/ \
    && python -m zipfile -e /wheels/gridoxide-*.whl /tmp/gx \
    && LD_LIBRARY_PATH=/opt/cudss/lib:/usr/local/cuda/lib64 ldd /tmp/gx/gridoxide/*.so /opt/cudss/lib/libcudss.so.1 \
       | awk '$3 ~ "^/usr/local/cuda" {print $3}' | sort -u | xargs -r cp -L -t /opt/cuda-rt/

FROM grid-bench/base:latest
WORKDIR /env
COPY tool-configs/gridoxide_gpu/pyproject.toml tool-configs/gridoxide_gpu/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --python 3.13.14
COPY --from=build /opt/cuda-rt /opt/cuda-rt
RUN echo /opt/cuda-rt > /etc/ld.so.conf.d/cuda-rt.conf && ldconfig
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    uv pip install --python /env/.venv/bin/python --no-deps /wheels/*.whl
ENV PATH="/env/.venv/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/gridoxide_gpu
WORKDIR /bench
