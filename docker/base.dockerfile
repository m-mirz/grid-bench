# Shared base for every grid-bench image: Debian + uv + a uv-managed CPython.
# No source is baked in: the repository is mounted read-only at /bench at run
# time, so editing an adapter or the oracle never requires a rebuild.
# Everything this image fetches is pinned: the base image and uv by digest,
# CPython by exact version (uv verifies its download against the checksum it
# ships for that version). No apt packages: uv verifies TLS with its bundled
# root certificates, and the tool containers have no network at run time.
FROM debian:bookworm-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171

COPY --from=ghcr.io/astral-sh/uv:0.11.33@sha256:77280f2f771df71f90786c314fe1bbc1e023feac652969bbf139c280babf2eb7 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PYTHONPATH=/bench \
    PYTHONDONTWRITEBYTECODE=1 \
    GRID_BENCH_RESULTS=/output \
    HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib
RUN uv python install 3.13.14

WORKDIR /bench
CMD ["bash"]
