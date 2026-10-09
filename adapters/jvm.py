"""The JVM of a Java CIM library (OpenCGMES, PowSyBl), started once in this
process through JPype, so the library is called in-process like every other
tool and its memory is this process's.

Started when the adapter is created, so that the memory baseline
(`adapters/memory.py`) includes the JVM and a case's import peak does not.
Both Java tools get the same heap limit, `HEAP`: room for RealGrid in either
triplestore on a 16 GB machine. Peak RSS includes whatever of the heap the
JVM has taken; that is the honest number for a library used through a
bridge. Version strings come from the jar names in `GRID_BENCH_JARS` (the
image's resolved, checksummed jars).
"""
import os
from pathlib import Path

HEAP = "8g"


def start() -> None:
    import jpype
    import jpype.imports  # noqa: F401 - Java packages importable as Python modules
    if not jpype.isJVMStarted():
        jpype.startJVM(f"-Xmx{HEAP}", classpath=[f"{os.environ['GRID_BENCH_JARS']}/*"], convertStrings=True)


def jar_version(artifact: str) -> str:
    """`cimxml` -> "1.3.0", from `cimxml-1.3.0.jar`."""
    jars = [p.name for p in Path(os.environ["GRID_BENCH_JARS"]).glob(f"{artifact}-*.jar")]
    found = [j.removeprefix(f"{artifact}-").removesuffix(".jar") for j in jars
             if j.removeprefix(f"{artifact}-")[:1].isdigit()]
    assert len(found) == 1, f"{artifact}: {jars}"
    return found[0]


def dependencies(*artifacts: str) -> dict[str, str]:
    import jpype
    from importlib.metadata import version
    return {"jpype1": version("jpype1"), "java": str(jpype.java.lang.System.getProperty("java.version")),
            **{a: jar_version(a) for a in artifacts}}
