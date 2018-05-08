# jackknife.py - although might want to rename this to statistical_analysis.py?

# system imports
import multiprocessing as mp
# import itertools as it
# import collections
# import subprocess
# import socket
# import glob
# import json
# import sys
# import os

# third party imports
import numpy as np

# local imports
# from .data import vibronic_model_io as vIO
from .data import postprocessing as pp
# from .data import file_structure
# from .data import file_name  # do we need this?
from .pimc.minimal import BoxResult, BoxResultPM
from . import constants
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
    delta_beta, rho, g, g_plus, g_minus, alpha_plus, alpha_minus = args

    ratio = g / rho

    LN = g_plus / rho
    LD = alpha_minus
    RN = g_minus / rho
    RD = alpha_minus

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


def calculate_alpha_jackknife_terms(terms, constants):
    """x"""
    # LN is g_plus / rho
    # RN is g_minus over / rho
    X, delta_beta, ratio, LN, RN, alpha_plus, alpha_minus = terms

    # first start with the full sum
    jk_ratio = np.full(shape=X, fill_value=np.sum(ratio))

    # only this needs special handling
    jk_LN = np.full(shape=X, fill_value=np.sum(LN))
    # jk_LD = np.full(shape=X, fill_value=np.sum(LD))
    jk_RN = np.full(shape=X, fill_value=np.sum(RN))
    # jk_RD = np.full(shape=X, fill_value=np.sum(RD))

    # now subtract each term from the sum
    jk_ratio -= ratio
    jk_LN -= LN
    # jk_LD -= LD
    jk_RN -= RN
    # jk_RD -= RD

    # pre normalize
    jk_ratio /= (X - 1)
    jk_LN /= (X - 1)
    # jk_LD /= (X - 1)
    jk_RN /= (X - 1)
    # jk_RD /= (X - 1)

    # now build the first and second symmetric derivative
    jk_sym_d1 = jk_LN * alpha_plus
    jk_sym_d1 -= jk_RN * alpha_minus
    jk_sym_d1 /= (2. * delta_beta)  # constant factor
    #
    jk_sym_d2 = jk_LN * alpha_plus
    jk_sym_d2 -= 2. * jk_ratio
    jk_sym_d2 += jk_RN * alpha_minus
    jk_sym_d2 /= pow(delta_beta, 2)  # constant factor

    ret = [jk_ratio,
           jk_sym_d1,
           jk_sym_d2]

    return ret


def calculate_jackknife_term(length, array):
    """calculate the jackknife term for a given array"""

    # start with the full sum
    jk_term = np.full(shape=length, fill_value=np.sum(array))
    # subtract each term from the sum
    jk_term -= array
    # normalize
    jk_term /= length

    return jk_term


def calculate_jackknife_terms(length, list_of_arrays):
    return [calculate_jackknife_term(length, array) for array in list_of_arrays]


def estimate_property(*args):
    """"""
    X, T, g_r, sym1, sym2 = args

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
    X, T, delta_beta, input_dict, jk_g_rho, jk_sym1, jk_sym2 = args

    # # Energy - jackknife edition
    # the jackknife estimator
    f_EJ = -1. * jk_sym1 / jk_g_rho
    #
    E = X * input_dict["E"]
    E -= (X - 1.) * np.mean(f_EJ)
    # error bars
    E_err = np.sqrt(X - 1.) * np.std(f_EJ, ddof=1)

    # # Heat Capacity - jackknife edition
    # the jackknife estimator
    f_CJ = jk_sym2 / jk_g_rho
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


# doesn't necessarily need the # of beads
# B is only needed for small 2x2 cases where we can do exact diagonalization
def perform_statistical_analysis(FS, X, P, T, B):
    """x"""
    pimc_results = BoxResultPM()
    rhoData = {}

    pp.load_data(X, P, B, T, pimc_results, rhoData)

    Z_sampling = rhoData["Z"]
    E_sampling = rhoData["E"]
    Cv_sampling = rhoData["Cv"]
    alpha_plus = rhoData["alpha_plus"]
    alpha_minus = rhoData["alpha_minus"]

    # byCopy?, should be byRef, double check this
    rho = pimc_results.s_rho
    g = pimc_results.s_g
    # rho_plus = pimc_results.s_rhop
    # rho_minus = pimc_results.s_rhom
    g_plus = pimc_results.s_gp
    g_minus = pimc_results.s_gm

    data = [rho, g, g_plus, g_minus]
    data_alpha = [rho, g, g_plus, g_minus, alpha_plus, alpha_minus]

    terms = calculate_property_terms(constants.delta_beta, data)
    terms_alpha = calculate_alpha_terms(constants.delta_beta, data_alpha)

    assert(np.allclose(terms[0], terms_alpha[0]))

    # Calculate the jackknife terms
    # JK_terms = calculate_jackknife_terms(X, terms)
    JK_terms_alpha = calculate_alpha_jackknife_terms(X, constants.delta_beta, g / rho, g_plus / rho, g_minus / rho, alpha_plus, alpha_minus)

    # calculate <exp> s.t. (H = <exp>)
    # ret = estimate_property(X, T, terms)
    # JK_ret = estimate_jackknife(X, T, constants.delta_beta, ret, JK_terms)

    # calculate <exp> s.t. (H = <exp> + ho)
    ret_alpha = estimate_property(terms_alpha, constants)
    JK_ret_alpha = estimate_jackknife(JK_terms_alpha, constants, ret_alpha)

    # Remember that we need to handle the difference terms specially
    add_harmonic_contribution(ret_alpha, E_sampling, Cv_sampling)
    add_harmonic_contribution(JK_ret_alpha, E_sampling, Cv_sampling)

    # we need to save data to file here!
    # output_path = path_results + filePrefix
    # output_path += "X{X:d}_P{P:d}_T{T:d}_thermo".format(X=X, P=P, T=T)
    # save to thermo file
    return


if (__name__ == "__main__"):

    # catalogue available files
    pimcList, coupledList, samplingList = pp.retrive_file_paths_for_jackknife()

    # find shared values
    arg_dict = pp.extract_jackknife_parameters(pimcList, coupledList, samplingList)

    # would be nice to have some feedback about what values there are and what are missing

    # manually select specific values from those available
    pimc_restriction = range(12, 101, 1)  # at least 12 beads before we plot
    # pick the highest number of samples
    sample_restriction = arg_dict["samples"][-1]
    # pick the highest number of basis functions
    basis_restriction = arg_dict["basis_fxns"][-1]
    # temperature is currently fixed at 300K
    temperature_restriction = np.array([300.00])

    # intersect returns sorted, unique values that are in both of the input arrays
    arg_dict["temperatures"] = np.intersect1d(arg_dict["temperatures"], temperature_restriction)
    arg_dict["pimc_beads"] = np.intersect1d(arg_dict["pimc_beads"], pimc_restriction)
    arg_dict["basis_fxns"] = np.intersect1d(arg_dict["basis_fxns"], basis_restriction)
    arg_dict["samples"] = np.intersect1d(arg_dict["samples"], sample_restriction)

    # create a list of arguments for multiprocessing pool
    arg_list = [(X, P, T, B)
                for X in arg_dict["samples"]
                for P in arg_dict["pimc_beads"]
                for T in arg_dict["temperatures"]
                for B in arg_dict["basis_fxns"]
                ]

    arg_iterator = iter(arg_list)

    block_size = 10

    with mp.Pool(block_size) as p:
        p.starmap(perform_statistical_analysis, arg_list)

    print("Finished")
