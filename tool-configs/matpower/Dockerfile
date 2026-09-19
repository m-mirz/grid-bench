# The matpower image: MATPOWER 8.1 on GNU Octave 11.3.0, plus the Python
# harness that drives it. Built by docker/build.sh in place of
# docker/tool.dockerfile.
#
# The official GNU Octave image is the base (pinned by digest; an OS package
# install is not allowed here), so the Python part repeats docker/base.dockerfile
# on top of it: uv by digest, CPython by exact version, packages by uv.lock.
# MATPOWER is the release zip, checked against its SHA-256.
FROM gnuoctave/octave:11.3.0@sha256:b22abbe607c15ed821521ab9cf8a7c466dc47506730a0de9503bd554731b3e32

COPY --from=ghcr.io/astral-sh/uv:0.11.33@sha256:77280f2f771df71f90786c314fe1bbc1e023feac652969bbf139c280babf2eb7 /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PYTHONPATH=/bench \
    PYTHONDONTWRITEBYTECODE=1 \
    GRID_BENCH_RESULTS=/output \
    HOME=/tmp
RUN uv python install 3.13.14

ADD --checksum=sha256:7f13b1441669a64e312d14a60e564cd91977ff1676ff77d25538e94ff313dd56 \
    https://github.com/MATPOWER/matpower/releases/download/8.1/matpower8.1.zip /tmp/matpower8.1.zip
RUN octave-cli --quiet --eval "unzip('/tmp/matpower8.1.zip', '/opt')" && rm /tmp/matpower8.1.zip \
    && chmod -R a+rX /opt/matpower8.1

WORKDIR /env
COPY tool-configs/matpower/pyproject.toml tool-configs/matpower/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --python 3.13.14
ENV PATH="/env/.venv/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/matpower
WORKDIR /bench
CMD ["bash"]
