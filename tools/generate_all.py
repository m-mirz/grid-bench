"""Regenerates every report from the result JSON in one directory.

    python -m tools.generate_all results-docker

Writes `docs/index.html`. GPU results live in `<dir>/gpu/`, with the CPU
baselines run on the same machine, and are shown apart from the rest (which
may come from another machine).
"""
import sys
from pathlib import Path

from tools import generate_site
from tools.benchmark_data import load

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main(directory: str) -> None:
    directory = Path(directory)
    res = load(directory)
    if not res.records and not res.failures:
        sys.exit(f"no grid-bench results in {directory}")
    DOCS.mkdir(exist_ok=True)
    gpu = load(directory / "gpu") if (directory / "gpu").is_dir() else None
    (DOCS / "index.html").write_text(generate_site.generate(directory, res, gpu))
    print(f"wrote {DOCS / 'index.html'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results-docker")
