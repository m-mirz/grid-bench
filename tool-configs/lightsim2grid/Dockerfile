# The lightsim2grid image: the shared base plus lightsim2grid compiled from
# its 1.1.0 sdist with p3s's compiler and flags. Built by docker/build.sh in
# place of docker/tool.dockerfile.
#
# p3s, the other C++/KLU Newton solver here, is compiled in its image
# (tool-configs/p3s/Dockerfile) by gcc 12 with its release's -O3
# -funroll-loops -ffast-math and no -march=native; the PyPI wheel of
# lightsim2grid is compiled elsewhere, by another compiler, with -O3 only.
# Built the same way, the two columns differ by their code, not their build.
# Pinned like everything else: the same gcc image as p3s by digest (gcc 12
# on bookworm, the base image's Debian, so the extension needs no newer
# glibc), the sdist by SHA-256 (the one uv.lock records), the build tools
# (cmake, ninja, scikit-build-core, pybind11, numpy) by uv.lock's `build`
# group. No apt packages.
FROM gcc:12-bookworm@sha256:c18df6babaae3d61789651ba987f767d05bcba481160301963673729d62666fa AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.33@sha256:77280f2f771df71f90786c314fe1bbc1e023feac652969bbf139c280babf2eb7 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.13.14
WORKDIR /env
COPY tool-configs/lightsim2grid/pyproject.toml tool-configs/lightsim2grid/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --only-group build --python 3.13.14
ENV PATH="/env/.venv/bin:${PATH}"

# lightsim2grid 1.1.0 from its sdist (SuiteSparse and Eigen included), with KLU.
# - CMake's Release build type (scikit-build-core's default): -O3 on every
#   target, KLU included, as SuiteSparse is built in the p3s image.
# - CXXFLAGS: p3s's -funroll-loops -ffast-math on lightsim2grid's C++ code;
#   in p3s they apply to its own C++ only, not to KLU (C), and here too.
# - __O3_OPTIM=1: -O3 on the Python bindings as well (lightsim2grid's own
#   switch; its core is -O3 already).
# - No __COMPILE_MARCHNATIVE: a portable build, like p3s (P3S_CPP_NATIVE=OFF).
# - libstdc++ linked dynamically, unlike p3s's setting: a shared-linker flag
#   reaches liblightsim2grid_core but not the Python module (a CMake MODULE,
#   which takes CMAKE_MODULE_LINKER_FLAGS), and the two exchange C++ objects,
#   so they must share one libstdc++: bookworm's, from the same gcc 12. (The
#   flag does not reach p3s's module either; it only affects portability.)
ADD --checksum=sha256:c24dbb9fe15699a84bd993efb4f86ea35fd533b3f18c941298f585e40669e771 \
    https://files.pythonhosted.org/packages/af/d5/4ce31c6bedc85adbd9a13fe0fd876f18727c3ce1485b20b979880a889762/lightsim2grid-1.1.0.tar.gz /src/lightsim2grid.tar.gz
RUN mkdir /src/lightsim2grid && tar -xzf /src/lightsim2grid.tar.gz -C /src/lightsim2grid --strip-components=1 \
    && __O3_OPTIM=1 CXXFLAGS="-funroll-loops -ffast-math" \
       uv build --wheel --no-build-isolation --python /env/.venv/bin/python -o /wheels /src/lightsim2grid

FROM grid-bench/base:latest
WORKDIR /env
COPY tool-configs/lightsim2grid/pyproject.toml tool-configs/lightsim2grid/uv.lock ./
# Everything the lockfile pins except lightsim2grid's PyPI wheel, then the
# wheel compiled above, of the same release (its one dependency, numpy, is
# in the lockfile's environment).
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --python 3.13.14 --no-install-package lightsim2grid
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    uv pip install --python /env/.venv/bin/python --no-deps /wheels/lightsim2grid-1.1.0-*.whl \
    && /env/.venv/bin/python -c "from lightsim2grid.algorithm import AlgorithmType; AlgorithmType.NR_KLU"
ENV PATH="/env/.venv/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/lightsim2grid
WORKDIR /bench
