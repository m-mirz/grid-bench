# Apache Jena Fuseki image for cgmes2pgm

Copied unchanged from `cgmes2pgm_suite` 0.4.3 (PyPI), directory
`cgmes2pgm_suite/resources/docker/`: the Fuseki image that suite builds and
starts for itself (Jena 5.5.0, in-memory datasets, `shiro.ini` allowing
anonymous access). cgmes2pgm reads CGMES only through a SPARQL endpoint, so
the benchmark runs this image as a sidecar of the `cgmes2pgm` service on an
internal network (see `docker/docker-compose.yml`).

Licence: Apache-2.0 (`LICENSE`, `NOTICE`); derived from the Apache Jena
Fuseki Dockerfile, modifications by SOPTIM AG as noted in the Dockerfile.

## Changes made here

For reproducible builds (everything a build fetches is pinned):
- both base images pinned by digest;
- the Fuseki jar checked against a pinned SHA-256 (taken from a jar whose
  Apache PGP signature verified with the Jena KEYS file and whose SHA-1
  matched Maven Central's), fetched with busybox `wget`;
- `apk add curl binutils` removed (curl replaced by busybox wget; jlink does
  not need binutils for `--strip-debug`), and with it `download.sh`, which
  checked the jar against a checksum fetched from the same server.
