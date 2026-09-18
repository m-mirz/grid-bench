# Apache Jena Fuseki image for cgmes2pgm

Copied unchanged from `cgmes2pgm_suite` 0.4.3 (PyPI), directory
`cgmes2pgm_suite/resources/docker/`: the Fuseki image that suite builds and
starts for itself (Jena 5.5.0, in-memory datasets, `shiro.ini` allowing
anonymous access). cgmes2pgm reads CGMES only through a SPARQL endpoint, so
the benchmark runs this image as a sidecar of the `cgmes2pgm` service on an
internal network (see `docker/docker-compose.yml`).

Licence: Apache-2.0 (`LICENSE`, `NOTICE`); derived from the Apache Jena
Fuseki Dockerfile, modifications by SOPTIM AG as noted in the Dockerfile.
