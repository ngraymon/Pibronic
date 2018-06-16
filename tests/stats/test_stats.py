""" """

# system imports
import inspect
import math
import os


# local imports
from ..context import pibronic
import pibronic.stats as st
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


def test_calculate_alpha_terms():
    """obviously not a good test at the moment"""
    X = 100
    delta_beta = 1.0
    rho = np.ones(X)
    g = np.ones(X)
    g_plus = np.ones(X) + 1
    g_minus = np.ones(X)
    alpha_plus = 2.0
    alpha_minus = 2.0

    ret = st.calculate_alpha_terms(delta_beta, rho, g, g_plus, g_minus, alpha_plus, alpha_minus)

    assert np.all(ret[0] == 1.0)
    assert ret[1] == 1.0
    assert ret[2] == 4.0
    assert np.all(ret[3] == 2.0)
    assert ret[4] == alpha_plus
    assert np.all(ret[5] == 1.0)
    assert ret[6] == alpha_minus
    return


def test_estimate_basic_properties():
    """obviously not a good test at the moment"""
    X = 100
    T = 300.00
    g_r = np.ones(X)
    sym1 = np.ones(X)
    sym2 = np.ones(X) + 1

    ret = st.estimate_basic_properties(X, T, g_r, sym1, sym2)
    kBT = boltzman * pow(T, 2.)

    assert ret["Z"] == 1.0
    assert ret["Z error"] == 0.0
    assert ret["E"] == -1.0
    assert ret["E error"] == 0.0
    assert ret["Cv"] == 1.0 / kBT
    assert ret["Cv error"] == 0.0
    return


def test_add_harmonic_contribution():
    nums = np.random.randint(0, 1000, size=4)
    test_dict = {"E": nums[0],"Cv": nums[1]}

    E_harmonic = nums[2]
    Cv_harmonic = nums[3]

    st.add_harmonic_contribution(test_dict, E_harmonic, Cv_harmonic)
    assert test_dict["E"] == nums[0] + nums[2]
    assert test_dict["Cv"] == nums[1] + nums[3]
    return


@pytest.fixture()
def path():
    # does this work when deployed?
    return os.path.join(os.path.dirname(os.path.dirname(inspect.getfile(pibronic))), "tests/test_models/")


@pytest.fixture(params=[(0, 0), (0, 1), (1, 0), (1, 1)])
def FS(path, request):
    return fs.FileStructure(path, request.param[0], id_rho=request.param[1])
