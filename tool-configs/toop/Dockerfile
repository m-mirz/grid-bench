# The ToOp image: the shared base, this lockfile's environment (JAX with
# NVIDIA's CUDA 12 wheels) and ToOp's sources. Built by docker/build.sh in
# place of docker/tool.dockerfile.
#
# ToOp is not on PyPI and has no release, so its sources come from a commit
# of its main branch, pinned by SHA-256: 2190b1b (2026-10-02), the newest at
# least 7 days old when it was added. Of its six packages, the four the DC
# solver imports are put on the path as they are (pure Python, no build
# step: its packaging derives versions from git tags, which a tarball has
# not). Their third-party dependencies are in pyproject.toml. No apt packages.
FROM grid-bench/base:latest
WORKDIR /env
COPY tool-configs/toop/pyproject.toml tool-configs/toop/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --python 3.13.14
ADD --checksum=sha256:9c90a2ddf5fcda08a9ae7e6d7d045671d592e4ce5b701d9b6184c5f18b502493 \
    https://github.com/eliagroup/ToOp/archive/2190b1b1393f6730d917e223543bf12ee48f87b4.tar.gz /src/toop.tar.gz
RUN mkdir -p /opt/toop /src/toop && tar -xzf /src/toop.tar.gz -C /src/toop --strip-components=1 \
    && for p in dc_solver interfaces grid_helpers contingency_analysis; do \
         cp -r /src/toop/packages/${p}_pkg/src/toop_engine_${p} /opt/toop/; done \
    && rm -rf /src/toop /src/toop.tar.gz \
    && echo /opt/toop > "$(/env/.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/toop.pth"
ENV PATH="/env/.venv/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/toop \
    TOOP_COMMIT=2190b1b1393f6730d917e223543bf12ee48f87b4
WORKDIR /bench
