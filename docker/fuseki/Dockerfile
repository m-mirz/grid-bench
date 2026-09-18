## Licensed to the Apache Software Foundation (ASF) under one or more
## contributor license agreements.  See the NOTICE file distributed with
## this work for additional information regarding copyright ownership.
## The ASF licenses this file to You under the Apache License, Version 2.0
## (the "License"); you may not use this file except in compliance with
## the License.  You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

## Apache Jena Fuseki server Dockerfile.

# Modifications copyright (C) 2025 SOPTIM AG
# Modified on 2025‑07‑21: Use shiro.ini to configure authentication. Update to Jena 5.5.0

## This Dockefile builds a reduced footprint container.

ARG JAVA_VERSION=21

ARG ALPINE_VERSION=3.21.2
ARG JENA_VERSION="5.5.0"

# Internal, passed between stages.
ARG FUSEKI_DIR=/fuseki
ARG FUSEKI_JAR=jena-fuseki-server-${JENA_VERSION}.jar
ARG JAVA_MINIMAL=/opt/java-minimal

## ---- Stage: Download and build java.
# Pinned by digest (grid-bench); the tag alone can move.
FROM eclipse-temurin:${JAVA_VERSION}-alpine@sha256:cd87715a8d45cfaa42419207c64680234f62785c49055cccd20437b5c9018380 AS base

ARG JAVA_MINIMAL
ARG JENA_VERSION
ARG FUSEKI_DIR
ARG FUSEKI_JAR
ARG REPO=https://repo1.maven.org/maven2
ARG JAR_URL=${REPO}/org/apache/jena/jena-fuseki-server/${JENA_VERSION}/${FUSEKI_JAR}

RUN [ "${JENA_VERSION}" != "" ] || { echo -e '\n**** Set JENA_VERSION ****\n' ; exit 1 ; }
RUN echo && echo "==== Docker build for Apache Jena Fuseki ${JENA_VERSION} ====" && echo

## -- Download the jar file and check it against a pinned SHA-256 (grid-bench).
## The upstream download.sh fetched the checksum from the same server as the
## jar; this hash was taken from a jar whose Apache PGP signature verified
## (Andy Seaborne's code-signing key, in https://downloads.apache.org/jena/KEYS)
## and whose SHA-1 matched Maven Central's. busybox wget replaces the unpinned
## `apk add curl`.
ARG JAR_SHA256=c8611d326ac387f1e5a854f9c56571e5f3961b9ec945783c9249a1fa01e0f820
WORKDIR $FUSEKI_DIR
RUN wget -q -O "${FUSEKI_JAR}" "$JAR_URL" && \
    echo "${JAR_SHA256}  ${FUSEKI_JAR}" | sha256sum -c -

## -- Make reduced Java JDK

ARG JDEPS_EXTRA="jdk.crypto.cryptoki,jdk.crypto.ec"
RUN \
  JDEPS="$(jdeps --multi-release base --print-module-deps --ignore-missing-deps ${FUSEKI_JAR})"  && \
  jlink \
        --compress 2 --strip-debug --no-header-files --no-man-pages \
        --output "${JAVA_MINIMAL}" \
        --add-modules "${JDEPS},${JDEPS_EXTRA}"

ADD entrypoint.sh .
ADD log4j2.properties .

## ---- Stage: Build runtime
FROM alpine:${ALPINE_VERSION}@sha256:56fa17d2a7e7f168a043a2712e63aed1f8543aeafdcee47c58dcffe38ed51099

## Import ARGs
ARG JENA_VERSION
ARG JAVA_MINIMAL
ARG FUSEKI_DIR
ARG FUSEKI_JAR

COPY --from=base /opt/java-minimal /opt/java-minimal
COPY --from=base /fuseki /fuseki
COPY shiro.ini $FUSEKI_DIR/run/shiro.ini

WORKDIR $FUSEKI_DIR

ARG LOGS=${FUSEKI_DIR}/logs
ARG DATA=${FUSEKI_DIR}/databases

ARG JENA_USER=fuseki
ARG JENA_GROUP=$JENA_USER
ARG JENA_GID=1000
ARG JENA_UID=1000

# Run as this user
# -H : no home directory
# -D : no password
RUN addgroup -g "${JENA_GID}" "${JENA_GROUP}" && \
    adduser "${JENA_USER}" -G "${JENA_GROUP}" -s /bin/ash -u "${JENA_UID}" -H -D

RUN mkdir --parents "${FUSEKI_DIR}" && \
    chown -R $JENA_USER ${FUSEKI_DIR}

USER $JENA_USER

RUN \
    mkdir -p $LOGS && \
    mkdir -p $DATA && \
    chmod a+x entrypoint.sh

## Default environment variables.
ENV \
    JAVA_HOME=${JAVA_MINIMAL}           \
    JAVA_OPTIONS="-Xmx2048m -Xms2048m"  \
    JENA_VERSION=${JENA_VERSION}        \
    FUSEKI_JAR="${FUSEKI_JAR}"          \
    FUSEKI_DIR="${FUSEKI_DIR}"

EXPOSE 3030

ENTRYPOINT ["./entrypoint.sh" ]
CMD []
