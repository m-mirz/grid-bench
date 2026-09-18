"""The cimoxide MATPOWER -> CGMES converter must be exact, and the tool-free
checker that says so must catch a converter that is not.

The cases cover every construct the converter emits: off-nominal
transformers and transformer line charging (case118, case300), a negative
reactance (case300), phase shifters (case1354pegase), PV buses without an
online generator, offline generators, and generators on PQ buses
(case3120sp, case2848rte).
"""
import re

import pytest

from cases import matpower_to_cgmes
from cases.matpower import parse_m
from cases.registry import CASES
from oracle.cgmes_model import fidelity

EXACT = 1e-12


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    def convert(case):
        out = tmp_path_factory.mktemp(case)
        mpc = parse_m(CASES[case]["file"])
        return mpc, matpower_to_cgmes.write(matpower_to_cgmes.convert(mpc, case), out, case)
    return convert


@pytest.mark.parametrize("case", ["case14", "case300", "case1354pegase", "case3120sp", "case2848rte"])
def test_conversion_is_exact(converted, case):
    mpc, paths = converted(case)
    f = fidelity(mpc, paths)
    assert f["buses"] == f["buses_expected"]
    assert f["max_dy_rel"] < EXACT
    assert f["max_ds_mva"] < 1e-9
    assert f["setpoints_missing"] == f["setpoints_extra"] == 0
    assert f["max_dv_setpoint_pu"] < 1e-12


def test_topological_nodes_are_defined_in_tp(converted):
    """Guards the workaround for cimoxide writing TP TopologicalNodes as bare
    references (see matpower_to_cgmes._define_topological_nodes)."""
    _, paths = converted("case14")
    tp = next(p for p in paths if "_TP" in p.name).read_text()
    assert len(re.findall(r'<cim:TopologicalNode rdf:ID="_', tp)) == 14
    assert "<cim:IdentifiedObject.name>BUS-1</cim:IdentifiedObject.name>" in tp


def test_checker_catches_a_sign_flip(converted):
    """pandapower's CGMES importer drops the sign of a negative line
    reactance; the same change in a file must be caught."""
    mpc, paths = converted("case300")
    eq = next(p for p in paths if "_EQ" in p.name)
    text = eq.read_text()
    assert "<cim:ACLineSegment.x>-" in text
    eq.write_text(text.replace("<cim:ACLineSegment.x>-", "<cim:ACLineSegment.x>", 1))
    assert fidelity(mpc, paths)["max_dy_rel"] > 1e-3


def test_checker_catches_a_dropped_phase_shift(converted):
    mpc, paths = converted("case1354pegase")
    eq = next(p for p in paths if "_EQ" in p.name)
    eq.write_text(re.sub(r"(<cim:PhaseTapChangerTablePoint.angle>)[^<]+", r"\g<1>0", eq.read_text()))
    assert fidelity(mpc, paths)["max_dy_rel"] > 1e-6
