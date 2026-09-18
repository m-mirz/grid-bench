"""The MATPOWER -> power-grid-model converter, `gridoxide.matpower`.

It is a single pure-Python module (numpy/scipy only) inside the gridoxide
package, but gridoxide 0.0.2 publishes no Linux wheel for Python 3.13, so
installing the package means compiling its Rust extension. Instead, the
pinned sdist is downloaded once, verified against its PyPI sha256, and only
`python/gridoxide/matpower.py` is extracted and imported.
"""
import hashlib
import importlib.util
import io
import tarfile
import urllib.request

from cases.registry import CACHE

VERSION = "0.0.2"
SDIST_URL = ("https://files.pythonhosted.org/packages/source/g/gridoxide/"
             f"gridoxide-{VERSION}.tar.gz")
SDIST_SHA256 = "19f9ef42a6268415a0ef16bc15a77b0b539b9ababd974c6651f37bbb0061f60e"
MODULE = CACHE / "vendor" / f"gridoxide-{VERSION}-matpower.py"


def _fetch() -> None:
    data = urllib.request.urlopen(SDIST_URL, timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SDIST_SHA256:
        raise RuntimeError(f"gridoxide sdist sha256 {digest} != pinned {SDIST_SHA256}")
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        source = tar.extractfile(f"gridoxide-{VERSION}/python/gridoxide/matpower.py").read()
    MODULE.parent.mkdir(parents=True, exist_ok=True)
    MODULE.write_bytes(source)


def converter():
    """The `gridoxide.matpower` module."""
    if not MODULE.exists():
        _fetch()
    spec = importlib.util.spec_from_file_location("gridoxide_matpower", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
