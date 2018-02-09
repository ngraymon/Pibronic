# jackknife.py


# system imports
import os

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
from .pimc.minimal import BoxResult, BoxResultPM
from . import constants
from .constants import hbar


assert(len(sys.argv) == 3)
assert(sys.argv[1].isnumeric() and int(sys.argv[1]) >= 1)
assert(sys.argv[2].isnumeric() and int(sys.argv[1]) >= 0)

dir_workspace = "/work/ngraymon/pimc/"
id_data   = int(sys.argv[1])
dir_data  = "data_set_{:d}/".format(id_data)
id_rho    = int(sys.argv[2])
dir_rho   = "rho_{:d}/".format(id_rho)

directory_path = dir_workspace + dir_data
sampling_path = directory_path + dir_rho
assert(os.path.exists(sampling_path))

# some default file paths
path_sos     = directory_path + "parameters/"
path_params  = sampling_path + "parameters/"
path_results = sampling_path + "results/"
path_plots   = sampling_path + "plots/"

# read in the model parameters from the coupled and sampling models
number_of_modes, number_of_surfaces = vIO.get_nmode_nsurf_from_coupled_model(id_data)
rho_modes, rho_surfs = vIO.get_nmode_nsurf_from_sampling_model(id_data, id_rho)

# FILE_ROOT  =  "F{:d}".format(id_data)           + \
#                 "_A{:d}".format(number_of_surfaces) + \
#                 "_N{:d}".format(number_of_modes)    + \
#                 "_X{:d}_P{:d}_T{:d}"

filePrefix = "F{F:d}_A{A:d}_N{N:d}_".format(F=id_data, A=number_of_surfaces, N=number_of_modes)
fileSuffix = "X{X:d}_P{P:d}_T{T:d}_data_points.npz"
sosSuffix = "sos_B{B:d}.json"

def retrive_file_list():
    """find a list of all files that might be used in jackknife"""
    # retrive pimc files
    globPath = path_results + "F{:d}_*_data_points.npz".format(id_data)
    pimcList = [file for file in glob.glob(globPath)]

    # retrive coupled files
    globPath = path_sos + "sos_B*.json"
    coupledList = [file for file in glob.glob(globPath)]

    # retrive sampling files
    globPath = path_params + "sos_B*.json"
    samplingList = [file for file in glob.glob(globPath)]

    return pimcList, coupledList, samplingList


def extract_parameters(pimcList, coupledList, samplingList):
    """make a list of all parameters whose dependencies are satisfied this list will be used to check what data can be jackknifed"""

    # note that the way these splits are coded
    # will pose problems if the naming scheme for sos is changed in the future
    valuesDictionary = {"beads":0, "samples":0, "basis_fxns":0, "temperatures":0}

    cL = map(lambda path: int(path.split("_B")[1].split(".json")[0]), coupledList)
    sL = map(lambda path: int(path.split("_B")[1].split(".json")[0]), samplingList)

    # parse file paths to find shared #'s' of basis functions
    valuesDictionary["basis_fxns"] = list(set(cL) & set(sL))
    valuesDictionary["basis_fxns"].sort()
    # print(valuesDictionary["basis_fxns"])


    # parse file paths to find shared temperature values
    # for now we will leave this partially undeveloped
    tempL = map(lambda path: int(path.split("_T")[1].split("_data_points.npz")[0]), pimcList)
    valuesDictionary["temperatures"] = list(set(tempL))
    valuesDictionary["temperatures"].sort()
    # print(valuesDictionary["temperatures"])

    # parse file paths to find shared sample values
    xL = map(lambda path: int(path.split("_X")[1].split("_P")[0]), pimcList)
    valuesDictionary["samples"] = list(set(xL))
    valuesDictionary["samples"].sort()
    # print(valuesDictionary["samples"])

    # parse file paths to find pimc bead values
    pL = map(lambda path: int(path.split("_P")[1].split("_T")[0]), pimcList)
    valuesDictionary["pimc_beads"] = []
    for p in pL:
        if p not in valuesDictionary["pimc_beads"]:
            valuesDictionary["pimc_beads"].append(p)
    valuesDictionary["pimc_beads"].sort()
    # print(valuesDictionary["pimc_beads"])

    return valuesDictionary


def load_data(X, P, B, T, pimc_results, rhoArgs):
    """"""
    data_points_path = path_results + filePrefix
    data_points_path += fileSuffix.format(X=X, P=P, T=T)
    sos_path = path_params + sosSuffix.format(B=B)


    pimc_results.load_results(file_path)

    try:
        # with np.load(data_points_path, allow_pickle=True) as npz_file:
        #     if id_data in [1, 2, 3, 4]:
        #         pimcArgs["y_rho"] = npz_file["rho"]
        #         pimcArgs["y_g"] = npz_file["g_beta"]
        #     else:
        #         pimcArgs["y_rho"] = npz_file["y_rho"]
        #         pimcArgs["y_g"] = npz_file["y_g"]

        #     pimcArgs["y_gp"] = npz_file["y_gp"]
        #     pimcArgs["yp_gp"] = npz_file["yp_gp"]

        #     pimcArgs["y_gm"] = npz_file["y_gm"]
        #     pimcArgs["ym_gm"] = npz_file["ym_gm"]

        #     pimcArgs["y_rhop"] = npz_file["y_rhop"]
        #     pimcArgs["yp_rhop"] = npz_file["yp_rhop"]

        #     pimcArgs["y_rhom"] = npz_file["y_rhom"]
        #     pimcArgs["ym_rhom"] = npz_file["ym_rhom"]

        # for key, value in pimcArgs.items():
        #     assert(value.size == X)

        # if(np.any(pimcArgs["y_rhom"] == 0.0) or np.any(pimcArgs["ym_rhom"] == 0.0)):
        #     raise Error("")

        # y_rhom and ym_rhom are somehow 0?
        # print(y_rho[:10], y_rhop[:10], y_rhom[:10], yp_rhop[:10], ym_rhom[:10], y_g[:10], y_gp[:10], y_gm[:10], yp_gp[:10], ym_gm[:10],  sep="\n")

        with open(sos_path, "r") as rho_file:
            rho_dict = json.loads(rho_file.read())
            input_temp_index = rho_dict["temperature"].index(T)
            # make sure the temperature matches
            assert(T == rho_dict["temperature"][input_temp_index])

            rhoArgs["Z"] = rho_dict["Z_sampling"][input_temp_index]
            rhoArgs["E"] = rho_dict["E_sampling"][input_temp_index]
            rhoArgs["Cv"] = rho_dict["Cv_sampling"][input_temp_index]
            # print("Zh", rho_dict["Z_ho_analytical"][input_temp_index])
            # print("Zh+", rho_dict["Z_ho_analytical+beta"][input_temp_index])

            rhoArgs["alpha_plus"] = rhoArgs["Z"] / rho_dict["Z_sampling+beta"][input_temp_index]
            rhoArgs["alpha_minus"] = rhoArgs["Z"] / rho_dict["Z_sampling-beta"][input_temp_index]
            # print(alpha_plus, alpha_minus)
            # print(E_sampling, Cv_sampling)

        # print(Z_sampling, E_sampling, Cv_sampling, alpha_plus, alpha_minus)

    except OSError as err:
        print("Skipped {:} and {:}".format(data_points_path, sos_path))
        # skip if we cannot obtain all the necessary data
        return

    # We worked
    print(data_points_path)
    # print(sos_f_name)
    # print(alpha_plus)
    # print(np.mean(y_rhop / y_rho))
    # print(np.mean(y_rho / y_rhop))
    # print(alpha_minus)
    # print(np.mean(y_rhom / y_rho))
    # print(np.mean(y_rho / y_rhom))
    # print(y_gplus[0:5])
    # print(y_gplus[0:5]*alpha_plus)
    # print(y_gminus[0:5])
    # print(y_gminus[0:5]*alpha_minus)
    # print(y_gplus[0:5] - y_gminus[0:5])
    # print(y_gplus[0:5]*alpha_plus - y_gminus[0:5]*alpha_minus)
    return


def calculate_property_terms(data, constants):
    """calculate g_r, sym_d1, sym_d2 given the estimation of the exact property"""
    X, P, T, delta_beta, kB = constants
    y_rho, y_g, y_gp, y_gm = data

    # # Precalculate the 3 terms we will use
    ratio = y_g / y_rho

    first_symmetric_derivative = (y_gp - y_gm) / y_rho
    first_symmetric_derivative /= (2. * delta_beta)  # constant factor

    second_symmetric_derivative = (y_gp - (2. * y_g) + y_gm) / y_rho
    second_symmetric_derivative /= pow(delta_beta, 2)  # constant factor

    ret = [ratio,
           first_symmetric_derivative,
           second_symmetric_derivative]

    return ret


def calculate_difference_terms(data, constants):
    """calculate g_r, sym_d1, sym_d2 given the estimation of the difference"""
    X, P, T, delta_beta, kB = constants
    y_rho, y_rhop, yp_rhop, y_rhom, ym_rhom, y_g, y_gp, yp_gp, y_gm, ym_gm = data

    # # Precalculate the 3 terms we will use
    ratio = y_g / y_rho

    first_symmetric_derivative = (yp_gp / yp_rhop) - (ym_gm / ym_rhom)
    first_symmetric_derivative /= (2. * delta_beta)  # constant factor

    second_symmetric_derivative = (yp_gp / yp_rhop) - (2. * ratio) + (ym_gm / ym_rhom)
    second_symmetric_derivative /= pow(delta_beta, 2)  # constant factor

    ret = [ratio,
           first_symmetric_derivative,
           second_symmetric_derivative]

    return ret


def calculate_alpha_terms(data, constants, Uho):
    """calculate g_r, sym_d1, sym_d2 given the estimation of the difference + alpha"""
    X, P, T, delta_beta, kB = constants
    y_rho, y_rhop, yp_rhop, y_rhom, ym_rhom, y_g, y_gp, yp_gp, y_gm, ym_gm, ap, am = data
    # rho, g, y_gp, y_gm, yp_rhop, ym_rhom, ap, am,  = data

    

    # the current alpha method
    # # Precalculate the 3 terms we will use
    ratio = y_g / y_rho

    LN = y_gp / y_rho
    # LD = np.full(shape=X, fill_value=ap)
    LD = ap
    # print(ap, np.mean(LD))
    # LD = y_rho / y_rhop
    RN = y_gm / y_rho
    # RD = np.full(shape=X, fill_value=am)
    RD = am
    # print(am, np.mean(RD))
    # RD = y_rho / y_rhom
    # pr = np.mean(y_rho / y_rhop)
    # mr = np.mean(y_rho / y_rhom)

    # print((pr - mr) / (2. * delta_beta))
    # print((np.mean(y_rhom / y_rho) - np.mean(y_rhop / y_rho)) / (2. * delta_beta))
    # print(Uho)


    # print(np.mean(ratio) * Uho)
    # print(np.mean(ratio) * Uho * 2. * delta_beta)

    first_symmetric_derivative = np.mean(LN) * np.mean(LD)
    first_symmetric_derivative -= np.mean(RN) * np.mean(RD)
    # first_symmetric_derivative = np.mean(LN)
    # first_symmetric_derivative -= np.mean(RN)
    # first_symmetric_derivative = (y_gp - y_gm) / y_rho
    first_symmetric_derivative /= (2. * delta_beta)  # constant factor
    # print(first_symmetric_derivative)

    # first_symmetric_derivative += -1. * Uho * np.mean(ratio)
    # first_symmetric_derivative += np.mean(ratio) * Uho
    # print(first_symmetric_derivative)
    # first_symmetric_derivative -= np.mean(ratio) * Uho
    # first_symmetric_derivative += np.mean(ratio) * (pr - mr) / (2. * delta_beta)
    # print(first_symmetric_derivative)
    # first_symmetric_derivative += (2. * delta_beta * np.mean(ratio)) * -1. * (pr - mr)

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
    """"""
    X, P, T, delta_beta, kB = constants
    g_r, sym_d1, sym_d2, LN, LD, RN, RD = terms
    # print("L", LN,LD,RN,RD, sep="\n")

    # first start with the full sum
    jk_ratio = np.full(shape=X, fill_value=np.sum(g_r))

    # only this needs special handling
    jk_LN = np.full(shape=X, fill_value=np.sum(LN))
    # jk_LD = np.full(shape=X, fill_value=np.sum(LD))
    jk_RN = np.full(shape=X, fill_value=np.sum(RN))
    # jk_RD = np.full(shape=X, fill_value=np.sum(RD))

    # now subtract each term from the sum
    jk_ratio -= g_r
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
    jk_sym_d1 =  jk_LN * LD
    jk_sym_d1 -= jk_RN * RD
    jk_sym_d1 /= (2. * delta_beta)  # constant factor
    #
    jk_sym_d2 =  jk_LN * LD
    jk_sym_d2 -= 2. * jk_ratio
    jk_sym_d2 += jk_RN * RD
    jk_sym_d2 /= pow(delta_beta, 2)  # constant factor

    ret = [jk_ratio,
           jk_sym_d1,
           jk_sym_d2]

    return ret


def calculate_fourth_terms(data, constants):
    """calculate g_r, sym_d1, sym_d2 given the estimation of the difference in a new way"""
    X, P, T, delta_beta, kB = constants
    y_rho, y_rhop, yp_rhop, y_rhom, ym_rhom, y_g, y_gp, yp_gp, y_gm, ym_gm, ap, am = data

    # # Precalculate the 3 terms we will use
    ratio = y_g / y_rho

    LN = y_gp  / y_rho
    LD = y_rhop / y_rho
    RN = y_gm  / y_rho
    RD = y_rhom / y_rho

    first_symmetric_derivative =  np.mean(LN) / np.mean(LD)
    first_symmetric_derivative -= np.mean(RN) / np.mean(RD)
    first_symmetric_derivative /= (2. * delta_beta)  # constant factor
    # print(first_symmetric_derivative)

    second_symmetric_derivative =  np.mean(LN) / np.mean(LD)
    second_symmetric_derivative -= 2. * np.mean(ratio)
    second_symmetric_derivative += np.mean(RN) / np.mean(RD)
    second_symmetric_derivative /= pow(delta_beta, 2)  # constant factor
    # print(second_symmetric_derivative)

    ret = [ratio,
           first_symmetric_derivative,
           second_symmetric_derivative,
           LN, LD, RN, RD,
           ]

    return ret


def calculate_fourth_jackknife_terms(terms, constants):
    """calculate all the jackknife averages"""
    X, P, T, delta_beta, kB = constants
    g_r, sym_d1, sym_d2, LN, LD, RN, RD = terms
    # print("L", LN,LD,RN,RD, sep="\n")

    # first start with the full sum
    jk_ratio = np.full(shape=X, fill_value=np.sum(g_r))

    # only this needs special handling
    jk_LN = np.full(shape=X, fill_value=np.sum(LN))
    jk_LD = np.full(shape=X, fill_value=np.sum(LD))
    jk_RN = np.full(shape=X, fill_value=np.sum(RN))
    jk_RD = np.full(shape=X, fill_value=np.sum(RD))

    # now subtract each term from the sum
    jk_ratio -= g_r
    jk_LN -= LN
    jk_LD -= LD
    jk_RN -= RN
    jk_RD -= RD

    # pre normalize
    jk_ratio /= (X - 1)
    jk_LN /= (X - 1)
    jk_LD /= (X - 1)
    jk_RN /= (X - 1)
    jk_RD /= (X - 1)

    # now build the first and second symmetric derivative
    jk_sym_d1 =  jk_LN / jk_LD
    jk_sym_d1 -= jk_RN / jk_RD
    jk_sym_d1 /= (2. * delta_beta)  # constant factor
    #
    jk_sym_d2 =  jk_LN / jk_LD
    jk_sym_d2 -= 2. * jk_ratio
    jk_sym_d2 += jk_RN / jk_RD
    jk_sym_d2 /= pow(delta_beta, 2)  # constant factor

    ret = [jk_ratio,
           jk_sym_d1,
           jk_sym_d2]

    return ret


def calculate_jackknife_terms(terms, constants):
    """calculate all the jackknife averages"""
    X, P, T, delta_beta, kB = constants
    g_r, sym_d1, sym_d2 = terms

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


def estimate_property(terms, constants):
    """"""
    X, P, T, delta_beta, kB = constants
    # print("t", len(terms), terms, sep="\n")
    g_r, sym1, sym2, *args = terms
    # print("e", g_r, sym1, sym2, args, sep="\n")

    # Partition Function
    Z = np.mean(g_r)
    Z_err = np.std(g_r, ddof=0)
    Z_err /= np.sqrt(X - 1)

    # Energy
    E = -1. * np.mean(sym1) / np.mean(g_r)
    # error
    E_err = np.zeros_like(E) # can't be measured
    # this is just a lower bound on my error bars
    # E_err = np.std(sym1 / g_r, ddof=0)
    # E_err /= np.sqrt(X - 1) # remember that this is necessary!

    # Heat Capacity
    Cv = np.mean(sym2) / np.mean(g_r)
    Cv -= pow(E, 2.)
    Cv /= kB * pow(T, 2.)
    # error
    Cv_err = np.zeros_like(Cv) # can't be measured
    # this is just a lower bound on my error bars
    # old_Cv_err = np.std(old_Cv_err, ddof=0)
    # old_Cv_err /= np.sqrt(X - 1) # remember that this is necessary!

    # easy to access storage
    return_dictionary = {"Z": Z,   "Z error": Z_err,
                         "E": E,   "E error": E_err,
                         "Cv": Cv, "Cv error": Cv_err,
                         }
    return return_dictionary


def estimate_jackknife(terms, constants, input_dict):
    """"""
    X, P, T, delta_beta, kB = constants
    jk_f, jk_sym1, jk_sym2 = terms

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
    f_CJ /= kB * pow(T, 2.)
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


def postprocess_difference(input_dict, E_sampling, Cv_sampling):
    """"""
    # add the harmonic contribution to the energy
    input_dict["E"] += E_sampling

    # add the harmonic contribution to the heat capacity
    input_dict["Cv"] += Cv_sampling
    return


def main(X, P, T, B):

    pimc_results = BoxResultPM(X)
    rhoData = {}

    load_data(X, P, B, T, pimc_results, rhoData)

    Z_sampling = rhoData["Z"]
    E_sampling = rhoData["E"]
    Cv_sampling = rhoData["Cv"]
    alpha_plus = rhoData["alpha_plus"]
    alpha_minus = rhoData["alpha_minus"]

    y_rho = pimcData["y_rho"]
    y_g = pimcData["y_g"]
    y_rhop = pimcData["y_rhop"]
    y_rhom = pimcData["y_rhom"]
    yp_rhop = pimcData["yp_rhop"]
    ym_rhom = pimcData["ym_rhom"]
    y_gp = pimcData["y_gp"]
    y_gm = pimcData["y_gm"]
    yp_gp = pimcData["yp_gp"]
    ym_gm = pimcData["ym_gm"]


    # # # Begin to calculate properties
    # input data
    data = [y_rho, y_g, y_gp, y_gm]    #
    data_diff = [y_rho, y_rhop, yp_rhop, y_rhom, ym_rhom,
                 y_g, y_gp, yp_gp, y_gm, ym_gm]
    #
    data_alpha = data_diff + [alpha_plus, alpha_minus]
    #
    data_fourth = data_alpha
    # data_fourth = data_diff + [alpha_plus, alpha_minus, Z_sampling]
    #
    # constants = [X, P, T, DELTA_BETA, constants.boltzman]
    constants = [X, P, T, constants.delta_beta, constants.boltzman]

    # Precalculate
    terms = calculate_property_terms(data, constants)
    # g_r, sym1, sym2 = terms
    terms_diff = calculate_difference_terms(data_diff, constants)
    # g_r, sym1, sym2 = terms_diff
    terms_alpha = calculate_alpha_terms(data_alpha, constants, E_sampling)
    # print(terms_alpha, "\n\n")
    # g_r, sym1, sym2 = terms_delta
    terms_fourth = calculate_fourth_terms(data_fourth, constants)
    # print(terms_fourth, "\n\n")
    # g_r, sym1, sym2 = terms_fourth

    assert(np.allclose(terms[0], terms_diff[0]))
    assert(np.allclose(terms[0], terms_alpha[0]))
    assert(np.allclose(terms[0], terms_fourth[0]))

    # Calculate the jackknife terms
    JK_terms = calculate_jackknife_terms(terms, constants)
    # jk_f, jk_sym1, jk_sym2 = JK_terms
    JK_terms_diff = calculate_jackknife_terms(terms_diff, constants)
    # jk_f, jk_sym1, jk_sym2 = JK_terms_diff
    JK_terms_alpha = calculate_alpha_jackknife_terms(terms_alpha, constants)
    # jk_f, jk_sym1, jk_sym2 = JK_terms_alpha
    JK_terms_fourth = calculate_fourth_jackknife_terms(terms_fourth, constants)
    # jk_f, jk_sym1, jk_sym2 = JK_terms_fourth

    # calculate <exp> s.t. (H = <exp>)
    ret = estimate_property(terms, constants)
    # Z, Z_err, E, E_err, Cv, Cv_Err = ret
    JK_ret = estimate_jackknife(JK_terms, constants, ret)
    # E, E_err, Cv, Cv_Err = JK_ret

    # calculate <exp> s.t. (H = <exp> + ho)
    ret_diff = estimate_property(terms_diff, constants)
    # Z, Z_err, E, E_err, Cv, Cv_Err = ret_diff
    JK_ret_diff = estimate_jackknife(JK_terms_diff, constants, ret_diff)
    # E, E_err, Cv, Cv_Err = JK_ret_diff

    # calculate <exp> s.t. (H = <exp> + ho)
    ret_alpha = estimate_property(terms_alpha, constants)
    # Z, Z_err, E, E_err, Cv, Cv_Err = ret_alpha
    JK_ret_alpha = estimate_jackknife(JK_terms_alpha, constants, ret_alpha)
    # E, E_err, Cv, Cv_Err = JK_ret_diff

    # calculate <exp> s.t. (H = <exp> + ho)
    ret_fourth = estimate_property(terms_fourth, constants)
    # Z, Z_err, E, E_err, Cv, Cv_Err = ret_fourth
    JK_ret_fourth = estimate_jackknife(JK_terms_fourth, constants, ret_fourth)
    # E, E_err, Cv, Cv_Err = JK_ret_fourth

    # Remember that we need to handle the difference terms specially
    postprocess_difference(ret_diff, E_sampling, Cv_sampling)
    postprocess_difference(JK_ret_diff, E_sampling, Cv_sampling)
    postprocess_difference(ret_alpha, E_sampling, Cv_sampling)
    postprocess_difference(JK_ret_alpha, E_sampling, Cv_sampling)
    postprocess_difference(ret_fourth, E_sampling, Cv_sampling)
    postprocess_difference(JK_ret_fourth, E_sampling, Cv_sampling)

    # print(ret_alpha, "\n\n")
    # print(ret_fourth, "\n\n")

    output_path = path_results + filePrefix
    output_path += "X{X:d}_P{P:d}_T{T:d}_thermo".format(X=X, P=P, T=T)
    # # # Write output to file
    if False:
        fstr = "%-18.18s"+"%-+34.30E"
        # save the data to the thermo file
        X=[ # Partition Function
            fstr % ("Zf/Zh", ret["Z"]),
            fstr % ("Zf/Zh error", ret["Z error"]),
            "",
            # Energy!
            fstr % ("Orig E no JK", ret["E"]),
            fstr % ("Error", ret["E error"]),
            fstr % ("Orig E with JK", JK_ret["E"]),
            fstr % ("Error", JK_ret["E error"]),
            fstr % ("Diff E no JK", ret_diff["E"]),
            fstr % ("Error", ret_diff["E error"]),
            fstr % ("Diff E with JK", ret_diff["E"]),
            fstr % ("Error", JK_ret_diff["E error"]),
            fstr % ("Alpha E no JK", ret_alpha["E"]),
            fstr % ("Error", ret_alpha["E error"]),
            fstr % ("Alpha E with JK", JK_ret_alpha["E"]),
            fstr % ("Error", JK_ret_alpha["E error"]),
            fstr % ("Fourth E no JK", ret_fourth["E"]),
            fstr % ("Error", ret_fourth["E error"]),
            fstr % ("Fourth E with JK", JK_ret_fourth["E"]),
            fstr % ("Error", JK_ret_fourth["E error"]),
            "",
            # Heat capacity
            fstr % ("Orig Cv no JK", ret["Cv"]),
            fstr % ("Error", ret["Cv error"]),
            fstr % ("Orig Cv with JK", JK_ret["Cv"]),
            fstr % ("Error", JK_ret["Cv error"]),
            fstr % ("Diff Cv no JK", ret_diff["Cv"]),
            fstr % ("Error", ret_diff["Cv error"]),
            fstr % ("Diff Cv with JK", JK_ret_diff["Cv"]),
            fstr % ("Error", JK_ret_diff["Cv error"]),
            fstr % ("Alpha Cv no JK", ret_alpha["Cv"]),
            fstr % ("Error", ret_alpha["Cv error"]),
            fstr % ("Alpha Cv with JK", JK_ret_alpha["Cv"]),
            fstr % ("Error", JK_ret_alpha["Cv error"]),
            fstr % ("Fourth Cv no JK", ret_fourth["Cv"]),
            fstr % ("Error", ret_fourth["Cv error"]),
            fstr % ("Fourth Cv with JK", JK_ret_fourth["Cv"]),
            fstr % ("Error", JK_ret_fourth["Cv error"]),
        ]
        for a in X:
            print(a)
        # np.savetxt(output_path, X, fmt="%s")
    else:
        fstr = "%-+34.30E"
        # save the data to the thermo file
        np.savetxt(output_path,
            X=[ # Partition Function
                ret["Z"],
                ret["Z error"],
                # Energy!
                ret["E"],
                ret["E error"],
                JK_ret["E"],
                JK_ret["E error"],
                ret_diff["E"],
                ret_diff["E error"],
                JK_ret_diff["E"],
                JK_ret_diff["E error"],
                ret_alpha["E"],
                ret_alpha["E error"],
                JK_ret_alpha["E"],
                JK_ret_alpha["E error"],
                ret_fourth["E"],
                ret_fourth["E error"],
                JK_ret_fourth["E"],
                JK_ret_fourth["E error"],
                # Heat capacity
                ret["Cv"],
                ret["Cv error"],
                JK_ret["Cv"],
                JK_ret["Cv error"],
                ret_diff["Cv"],
                ret_diff["Cv error"],
                JK_ret_diff["Cv"],
                JK_ret_diff["Cv error"],
                ret_alpha["Cv"],
                ret_alpha["Cv error"],
                JK_ret_alpha["Cv"],
                JK_ret_alpha["Cv error"],
                ret_fourth["Cv"],
                ret_fourth["Cv error"],
                JK_ret_fourth["Cv"],
                JK_ret_fourth["Cv error"],
            ], fmt=fstr)
    return


if (__name__ == "__main__"):

    # catalogue available files
    pimcList, coupledList, samplingList = retrive_file_list()

    # find shared values
    arg_dict = extract_parameters(pimcList, coupledList, samplingList)

    # would be nice to have some feedback about what values there are and what are missing


    # manually select specific values from those available
    pimc_restriction = range(12, 101, 1) # at least 12 beads before we plot
    # pick the highest number of samples
    sample_restriction = arg_dict["samples"][-1]
    # pick the highest number of basis functions
    basis_restriction = arg_dict["basis_fxns"][-1]
    # temperature is currently fixed at 300K
    temperature_restriction = np.array([300])

    # intersect returns sorted, unique values that are in both of the input arrays
    arg_dict["temperatures"] = np.intersect1d(arg_dict["temperatures"], temperature_restriction)
    arg_dict["pimc_beads"] = np.intersect1d(arg_dict["pimc_beads"], pimc_restriction)
    arg_dict["basis_fxns"] = np.intersect1d(arg_dict["basis_fxns"], basis_restriction)
    arg_dict["samples"] = np.intersect1d(arg_dict["samples"], sample_restriction)

    # create a list of arguments for multiprocessing pool
    # arg_list = []
    # for X in arg_dict["samples"]:
    #     for P in arg_dict["pimc_beads"]:
    #         for T in arg_dict["temperatures"]:
    #             for B in arg_dict["basis_fxns"]:
    #                 arg_list.append((X, P, T, B))
    arg_list = [(X, P, T, B)                    \
                    for X in arg_dict["samples"]
                    for P in arg_dict["pimc_beads"]
                    for T in arg_dict["temperatures"]
                    for B in arg_dict["basis_fxns"]
                ]


    arg_iterator = iter(arg_list)


    block_size = 10

    with mp.Pool(block_size) as p:
        p.starmap(main, arg_list)

    print("Finished")


