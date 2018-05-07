# jackknife.py

# system imports
import multiprocessing as mp
import itertools as it
import collections
import subprocess
import socket
import glob
import json
import sys
import os

# third party imports
import numpy as np

# local imports
from .data import vibronic_model_io as vIO
from .data import postprocessing as pp
from .data import file_structure
from .data import file_name  # do we need this?
from .pimc.minimal import BoxResult, BoxResultPM
from .constants import boltzman


def calculate_property_terms(*args):
    """calculate g/rho, sym_d1, sym_d2 given the estimation of the exact property"""
    delta_beta, rho, g, g_plus, g_minus = args

    # # Precalculate the 3 terms we will use
    ratio = g / rho

    first_symmetric_derivative = (g_plus - g_minus) / rho
    first_symmetric_derivative /= (2. * delta_beta)  # constant factor

    second_symmetric_derivative = (g_plus - (2. * g) + g_minus) / rho
    second_symmetric_derivative /= pow(delta_beta, 2)  # constant factor

    ret = [ratio,
           first_symmetric_derivative,
           second_symmetric_derivative]

    return ret


def calculate_alpha_terms(*args):
    """calculate g/r, sym_d1, sym_d2 given the estimation of the difference + alpha"""
    delta_beta, y_rho, y_g, y_gp, y_gm, ap, am = args

    ratio = y_g / y_rho

    LN = y_gp / y_rho
    LD = ap
    RN = y_gm / y_rho
    RD = am

    first_symmetric_derivative = np.mean(LN) * np.mean(LD)
    first_symmetric_derivative -= np.mean(RN) * np.mean(RD)
    first_symmetric_derivative /= (2. * delta_beta)  # constant factor

    second_symmetric_derivative = np.mean(LN) * np.mean(LD)
    second_symmetric_derivative -= 2. * np.mean(ratio)
    second_symmetric_derivative += np.mean(RN) * np.mean(RD)
    second_symmetric_derivative /= pow(delta_beta, 2)  # constant factor

    ret = [ratio,
           first_symmetric_derivative,
           second_symmetric_derivative,
           LN, LD, RN, RD,
           ]

    return ret


def calculate_jackknife_terms(*args):
    """calculate all the jackknife averages"""
    X, delta_beta, g_r, sym_d1, sym_d2 = args

    # first start with the full sum
    jk_ratio = np.full(shape=X, fill_value=np.sum(g_r))
    jk_sym_d1 = np.full(shape=X, fill_value=np.sum(sym_d1))
    jk_sym_d2 = np.full(shape=X, fill_value=np.sum(sym_d2))

    # now subtract each term from the sum
    jk_ratio -= g_r
    jk_sym_d1 -= sym_d1
    jk_sym_d2 -= sym_d2

    # now normalize
    jk_ratio /= (X - 1)
    jk_sym_d1 /= (X - 1)
    jk_sym_d2 /= (X - 1)

    ret = [jk_ratio,
           jk_sym_d1,
           jk_sym_d2]

    return ret


def estimate_property(*args):
    """"""
    X, T, delta_beta, g_r, sym1, sym2 = args

    # Partition Function
    Z = np.mean(g_r)
    Z_err = np.std(g_r, ddof=0)
    Z_err /= np.sqrt(X - 1)

    # Energy
    E = -1. * np.mean(sym1) / np.mean(g_r)
    # error
    E_err = np.zeros_like(E)  # can't be measured
    # this is just a lower bound on my error bars
    # E_err = np.std(sym1 / g_r, ddof=0)
    # E_err /= np.sqrt(X - 1) # remember that this is necessary!

    # Heat Capacity
    Cv = np.mean(sym2) / np.mean(g_r)
    Cv -= pow(E, 2.)
    Cv /= boltzman * pow(T, 2.)
    # error
    Cv_err = np.zeros_like(Cv)  # can't be measured
    # this is just a lower bound on my error bars
    # old_Cv_err = np.std(old_Cv_err, ddof=0)
    # old_Cv_err /= np.sqrt(X - 1) # remember that this is necessary!

    # easy to access storage
    return_dictionary = {"Z": Z,   "Z error": Z_err,
                         "E": E,   "E error": E_err,
                         "Cv": Cv, "Cv error": Cv_err,
                         }
    return return_dictionary


def estimate_jackknife(*args):
    """"""
    X, T, delta_beta, input_dict, jk_f, jk_sym1, jk_sym2 = args

    # # Energy - jackknife edition
    # the jackknife estimator
    f_EJ = -1. * jk_sym1 / jk_f
    #
    E = X * input_dict["E"]
    E -= (X - 1.) * np.mean(f_EJ)
    # error bars
    E_err = np.sqrt(X - 1.) * np.std(f_EJ, ddof=1)

    # # Heat Capacity - jackknife edition
    # the jackknife estimator
    f_CJ = jk_sym2 / jk_f
    f_CJ -= pow(f_EJ, 2.)
    f_CJ /= boltzman * pow(T, 2.)
    #
    Cv = X * input_dict["Cv"]
    Cv -= (X - 1.) * np.mean(f_CJ)
    # error bars
    Cv_err = np.sqrt(X - 1.) * np.std(f_CJ, ddof=1)

    # easy to access storage
    return_dictionary = {"E": E,   "E error": E_err,
                         "Cv": Cv, "Cv error": Cv_err,
                         }
    return return_dictionary


def add_harmonic_contribution(input_dict, E_sampling, Cv_sampling):
    """"""
    # add the harmonic contribution to the energy
    input_dict["E"] += E_sampling

    # add the harmonic contribution to the heat capacity
    input_dict["Cv"] += Cv_sampling
    return


def perform_jackknife():

    terms = calculate_property_terms(data, constants)
    # g_r, sym1, sym2 = terms
    terms_alpha = calculate_alpha_terms(data_alpha, constants, E_sampling)
    # print(terms_alpha, "\n\n")
    # g_r, sym1, sym2 = terms_delta

    assert(np.allclose(terms[0], terms_alpha[0]))

    # Calculate the jackknife terms
    JK_terms = calculate_jackknife_terms(terms, constants)
    # jk_f, jk_sym1, jk_sym2 = JK_terms
    JK_terms_alpha = calculate_alpha_jackknife_terms(terms_alpha, constants)
    # jk_f, jk_sym1, jk_sym2 = JK_terms_alpha

    # calculate <exp> s.t. (H = <exp>)
    ret = estimate_property(terms, constants)
    # Z, Z_err, E, E_err, Cv, Cv_Err = ret
    JK_ret = estimate_jackknife(JK_terms, constants, ret)
    # E, E_err, Cv, Cv_Err = JK_ret

    # calculate <exp> s.t. (H = <exp> + ho)
    ret_alpha = estimate_property(terms_alpha, constants)
    # Z, Z_err, E, E_err, Cv, Cv_Err = ret_alpha
    JK_ret_alpha = estimate_jackknife(JK_terms_alpha, constants, ret_alpha)
    # E, E_err, Cv, Cv_Err = JK_ret_diff

    # Remember that we need to handle the difference terms specially
    postprocess_difference(ret_alpha, E_sampling, Cv_sampling)
    postprocess_difference(JK_ret_alpha, E_sampling, Cv_sampling)
    return


def perform_jackknife(FS, P, T, B):
    """x"""

    X = calculate_X(FS, P, T)

    pimc_results = BoxResultPM(X)
    rhoData = {}

    load_data(X, P, B, T, pimc_results, rhoData)

    Z_sampling = rhoData["Z"]
    E_sampling = rhoData["E"]
    Cv_sampling = rhoData["Cv"]
    alpha_plus = rhoData["alpha_plus"]
    alpha_minus = rhoData["alpha_minus"]

    y_rho = pimc_results.y_rho
    y_g = pimc_results.y_g
    y_rhop = pimc_results.y_rhop
    y_rhom = pimc_results.y_rhom
    y_gp = pimc_results.y_gp
    y_gm = pimc_results.y_gm

    data = [y_rho, y_g, y_gp, y_gm]
    data_alpha = [y_rho, y_rhop, y_rhom, y_g, y_gp, y_gm, alpha_plus, alpha_minus]

    # constants = [X, P, T, DELTA_BETA, constants.boltzman]
    constants = [X, P, T, constants.delta_beta, constants.boltzman]
    return
