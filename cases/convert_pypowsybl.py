"""MATPOWER case -> CGMES 3.0 through pypowsybl: its MATPOWER importer
followed by its CGMES exporter.

    python -m cases.convert_pypowsybl <case.mat> <out_dir>

The other converter the benchmark compares, against `cases/matpower_to_cgmes.py`
(cimoxide). Whatever pypowsybl's importer changes about the case (the
oracle already shows that it moves transformer line charging to one side)
carries into every tool that reads this output; `oracle.cgmes_model.fidelity`
measures it before any tool runs.

Settings: CIM 100 (CGMES 3.0); profiles EQ, TP, SSH (no SV: a published
solution would hand tools the answer); naming strategy `cgmes`, which gives
UUID mRIDs as CGMES requires while keeping IIDM bus-breaker bus ids as
TopologicalNode names (`BUS-<n>`, the MATPOWER bus number).
"""
import sys
from pathlib import Path


def convert(mat_path: Path, out_dir: Path) -> list[Path]:
    import pypowsybl.network as pn

    out_dir.mkdir(parents=True, exist_ok=True)
    network = pn.load(str(mat_path))
    network.save(str(out_dir / mat_path.stem), format="CGMES", parameters={
        "iidm.export.cgmes.cim-version": "100",
        "iidm.export.cgmes.profiles": "EQ,TP,SSH",
        "iidm.export.cgmes.naming-strategy": "cgmes",
    })
    return sorted(out_dir.glob("*.xml"))


if __name__ == "__main__":
    for p in convert(Path(sys.argv[1]), Path(sys.argv[2])):
        print(p)
