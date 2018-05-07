""" """

# system imports
import math

# third party imports
import numpy as np
import pibronic.jackknife as jk
from pibronic.constants import boltzman


def test_calculate_property_terms():
    """obviously not a good test at the moment"""
    X = 100
    delta_beta = 1.0
    rho = np.ones(X)
    g = np.ones(X)
    g_plus = np.ones(X) + 1
    g_minus = np.ones(X)

    ret = jk.calculate_property_terms(delta_beta, rho, g, g_plus, g_minus)

    assert np.all(ret[0] == g / rho)
    assert np.all(ret[1] == 0.5)
    assert np.all(ret[2] == 1.0)
    return


def test_calculate_jackknife_terms():
    """obviously not a good test at the moment"""
    X = 100
    delta_beta = 1.0
    rho = np.ones(X)
    g = np.ones(X)
    g_plus = np.ones(X) + 1
    g_minus = np.ones(X)

    ret = jk.calculate_property_terms(delta_beta, rho, g, g_plus, g_minus)

    assert np.all(ret[0] == g / rho)
    assert np.all(ret[1] == 0.5)
    assert np.all(ret[2] == 1.0)
    return


def test_estimate_property():
    """obviously not a good test at the moment"""
    X = 100
    T = 300.00
    delta_beta = 1.0
    g_r = np.ones(X)
    sym1 = np.ones(X)
    sym2 = np.ones(X) + 1

    ret = jk.estimate_property(X, T, delta_beta, g_r, sym1, sym2)
    kBT = boltzman * pow(T, 2.)

    assert ret["Z"] == 1.0
    assert ret["Z error"] == 0.0
    assert ret["E"] == -1.0
    assert ret["E error"] == 0.0
    assert ret["Cv"] == 1.0 / kBT
    assert ret["Cv error"] == 0.0
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


