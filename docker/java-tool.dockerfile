# The image of a CIM library on the JVM (opencgmes, powsybl), driven in-process
# through JPype. Built by docker/build.sh for every tool-configs/<TOOL>/ with
# a pom.xml, in place of docker/tool.dockerfile.
#
# Stage 1 resolves the pom's jars with Maven and checks every one against the
# committed jars.sha256 (and that there is no other): the Java side's
# lockfile, re-generated with docker/lock.sh. Stage 2 is the official JRE
# image plus the Python harness, as tool-configs/matpower/Dockerfile does on
# Octave: uv by digest, CPython by exact version, packages by uv.lock.
FROM maven:3.9.11-eclipse-temurin-21-noble@sha256:6fdc855a6ed81d288ca7ca37ac6ff5e9308b612485c0801d70b25a858c83d237 AS jars
ARG TOOL
WORKDIR /build
COPY tool-configs/${TOOL}/pom.xml tool-configs/${TOOL}/jars.sha256 ./
RUN --mount=type=cache,target=/root/.m2 \
    mvn -q -B dependency:copy-dependencies -DoutputDirectory=/build/lib -DincludeScope=runtime \
    && cd lib && sha256sum --strict -c ../jars.sha256 \
    && [ "$(ls | wc -l)" = "$(wc -l < ../jars.sha256)" ]

FROM eclipse-temurin:21-jre-noble@sha256:000fd431958bc81a24abe1e8e5f0f0fd3ae365a594bd50aadb20696805f9408c
ARG TOOL
COPY --from=ghcr.io/astral-sh/uv:0.11.33@sha256:77280f2f771df71f90786c314fe1bbc1e023feac652969bbf139c280babf2eb7 /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PYTHONPATH=/bench \
    PYTHONDONTWRITEBYTECODE=1 \
    GRID_BENCH_RESULTS=/output \
    GRID_BENCH_JARS=/opt/jars \
    HOME=/tmp
RUN uv python install 3.13.14
COPY --from=jars /build/lib /opt/jars

WORKDIR /env
COPY tool-configs/${TOOL}/pyproject.toml tool-configs/${TOOL}/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --python 3.13.14
ENV PATH="/env/.venv/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/${TOOL}
WORKDIR /bench
CMD ["bash"]
