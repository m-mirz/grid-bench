# The p3s image: the shared base plus p3s's compiled C++/KLU Newton solver
# (nr_klu). Built by docker/build.sh in place of docker/tool.dockerfile.
#
# The Python package comes from PyPI through uv.lock like every other tool,
# but its C++ extension (the `p3s-cpp` distribution behind the `[cpp]` extra)
# is not published, so a builder stage compiles it from the same release's
# source. Pinned like everything else: the gcc image by digest (gcc 12 on
# bookworm, the base image's Debian, so the extension needs no newer glibc),
# the SuiteSparse and p3s source tarballs by SHA-256, the build tools
# (cmake, ninja, scikit-build-core, pybind11, numpy) by uv.lock's `build`
# group. No apt packages.
FROM gcc:12-bookworm@sha256:c18df6babaae3d61789651ba987f767d05bcba481160301963673729d62666fa AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.33@sha256:77280f2f771df71f90786c314fe1bbc1e023feac652969bbf139c280babf2eb7 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.13.14
WORKDIR /env
COPY tool-configs/p3s/pyproject.toml tool-configs/p3s/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --only-group build --python 3.13.14
ENV PATH="/env/.venv/bin:${PATH}"

# KLU and what it needs (AMD, COLAMD, BTF, SuiteSparse_config), static and
# position-independent, so the extension carries it and the runtime image
# needs no SuiteSparse. No CHOLMOD (KLU's optional ordering, not used by
# p3s), no BLAS, no OpenMP, no Fortran.
ADD --checksum=sha256:81e560e1f74546df139edb765b3f5bc865866da23062312ffe8fd821063c8397 \
    https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v7.14.1.tar.gz /src/suitesparse.tar.gz
RUN mkdir /src/suitesparse && tar -xzf /src/suitesparse.tar.gz -C /src/suitesparse --strip-components=1 \
    && cmake -S /src/suitesparse -B /src/suitesparse/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;colamd;btf;klu" \
       -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
       -DKLU_USE_CHOLMOD=OFF -DSUITESPARSE_USE_OPENMP=OFF -DSUITESPARSE_USE_FORTRAN=OFF \
       -DSUITESPARSE_REQUIRE_BLAS=OFF -DBLA_VENDOR=Generic -DSUITESPARSE_DEMOS=OFF -DCMAKE_INSTALL_PREFIX=/opt/suitesparse \
    && cmake --build /src/suitesparse/build && cmake --install /src/suitesparse/build

# nr_klu from the v1.1.0 tag, the release uv.lock installs the Python side of.
# - P3S_CPP_NATIVE=OFF: a portable build, like every other tool's wheel. The
#   release's pyproject.toml means to set this but spells it `p3s_CPP_NATIVE`,
#   which CMake (case-sensitive) ignores, so its default is -march=native.
# - OpenMP on: it parallelises `solve_batch` (many operating points), which
#   the batch benchmark times at 1..n threads (adapters/p3s_batch_adapter.py);
#   single solves never use it. Its runtime, libgomp, comes from this same
#   pinned gcc image (copied below), since the base image has none and takes
#   no apt packages.
# - libstdc++ and libgcc linked statically: the extension then needs only glibc.
# Upstream's -O3 -funroll-loops -ffast-math stay as the release ships them.
ADD --checksum=sha256:efbf57ea0fd8ffdb692a339f366751a70e3b940c75e10fc8e3e5655b1c7a2798 \
    https://github.com/e2nIEE/parallel-pandapower-solver/archive/refs/tags/v1.1.0.tar.gz /src/p3s.tar.gz
RUN mkdir /src/p3s && tar -xzf /src/p3s.tar.gz -C /src/p3s --strip-components=1 \
    && CMAKE_PREFIX_PATH=/opt/suitesparse uv build --wheel --no-build-isolation --python /env/.venv/bin/python -o /wheels /src/p3s/p3s/cpp \
       --config-setting=cmake.define.P3S_CPP_NATIVE=OFF \
       "--config-setting=cmake.define.CMAKE_SHARED_LINKER_FLAGS=-static-libstdc++ -static-libgcc"

FROM grid-bench/base:latest
WORKDIR /env
COPY tool-configs/p3s/pyproject.toml tool-configs/p3s/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --python 3.13.14
# OpenMP's runtime for the extension, from the gcc that compiled it.
COPY --from=build /usr/local/lib64/libgomp.so.1.0.0 /usr/local/lib/libgomp.so.1
RUN ldconfig
# The extension installs into the p3s package (`from p3s import nr_klu`); its
# one dependency, numpy, is already in the lockfile's environment.
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    uv pip install --python /env/.venv/bin/python --no-deps /wheels/*.whl \
    && /env/.venv/bin/python -c "from p3s import nr_klu"
ENV PATH="/env/.venv/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/p3s
WORKDIR /bench
