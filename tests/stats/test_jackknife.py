""" """

# system imports
import inspect
import math
import os


# local imports
from ..context import pibronic
import pibronic.stats as st
import pibronic.stats.jackknife as jk
import pibronic.data.file_structure as fs
from pibronic.constants import boltzman

# third party imports
import numpy as np
import pytest


def test_calculate_basic_property_terms():
    """obviously not a good test at the moment"""
    X = 100
    delta_beta = 1.0
    rho = np.ones(X)
    g = np.ones(X)
    g_plus = np.ones(X) + 1
    g_minus = np.ones(X)

    ret = st.calculate_basic_property_terms(delta_beta, rho, g, g_plus, g_minus)

    assert np.all(ret[0] == g / rho)
    assert np.all(ret[1] == 0.5)
    assert np.all(ret[2] == 1.0)
    return


def test_estimate_jackknife():
    """obviously not a good test at the moment"""
    X = 100
    T = 300.00
    delta_beta = 1.0
    jk_f = np.ones(X)
    jk_sym1 = np.ones(X)
    jk_sym2 = np.ones(X) + 1

    kBT = boltzman * pow(T, 2.)
    input_dict = {"E": -1.0, "Cv": 1.0 / kBT}

    ret = jk.estimate_jackknife(X, T, delta_beta, input_dict, jk_f, jk_sym1, jk_sym2)

    assert ret["E"] == -1.0
    assert ret["E error"] == 0.0
    assert math.isclose(ret["Cv"], 1.0 / kBT)
    assert math.isclose(ret["Cv error"], 2.7755575615628914e-16)
    return


@pytest.fixture()
def path():
    # does this work when deployed?
    return os.path.join(os.path.dirname(os.path.dirname(inspect.getfile(pibronic))), "tests/test_models/")


@pytest.fixture(params=[(0, 0), (0, 1), (1, 0), (1, 1)])
def FS(path, request):
    return fs.FileStructure(path, request.param[0], id_rho=request.param[1])
