"""
Handles all file processing after calculations are run

Provides functions for collating files for statistical analaysis (such as jackknife) and plotting
"""

# system imports
import multiprocessing as mp
import itertools as it
import collections
import subprocess
import socket
import json
import glob
import sys
import os

# third party imports
import numpy as np
from numpy import newaxis as NEW
from numpy import float64 as F64

# local imports
# from ..data import vibronic_model_io as vIO
from .. import constants
# from ..constants import hbar
from ..log_conf import log
from ..data import file_name
from ..data import file_structure
# from ..server import job_boss

# TODO - should these single file functions exist? is there enough need to justify their use?
# def retrive_a_pimc_file(files):
#     """verify that a specific pimc file exists and retrives the path to it"""
#     globPath = FS.path_rho_results + file_name.jackknife(P="*", T="*", X="*")
#     return glob.glob(globPath)
# def retrive_a_sos_coupled_file(files):
#     """verify that a specific sos(coupled) file exists and retrives the path to it"""
#     globPath = FS.path_vib_params + file_name.sos("*")
#     return glob.glob(globPath)
# def retrive_a_sos_sampling_file(files):
#     """verify that a specific sos(sampling) file exists and retrives the path to it"""
#     globPath = FS.path_rho_params + file_name.sos("*")
#     return glob.glob(globPath)


def retrive_pimc_file_list(FS):
    """return a list of the full path to each pimc file that might be used"""
    globPath = FS.path_rho_results + file_name.pimc(P="*", T="*", J="*")
    return [file for file in glob.glob(globPath)]


def retrive_jackknife_file_list(FS):
    """return a list of the full path to each jackknife file that might be used"""
    globPath = FS.path_rho_results + file_name.jackknife(P="*", T="*", X="*")
    return [file for file in glob.glob(globPath)]


def retrive_sos_coupled_file_list(FS, B="*"):
    """return a list of the full path to each sos(coupled) file that might be used"""
    globPath = FS.path_vib_params + file_name.sos("*")
    return [file for file in glob.glob(globPath)]


def retrive_sos_sampling_file_list(FS, B="*"):
    """return a list of the full path to each sos(sampling) file that might be used"""
    globPath = FS.path_rho_params + file_name.sos("*")
    return [file for file in glob.glob(globPath)]


def retrive_all_file_paths(FS):
    """return three lists of the full paths to each data file that might be used"""

    list_pimc = retrive_pimc_file_list(FS)
    list_jackknife = retrive_jackknife_file_list(FS)
    list_sos_vib = retrive_sos_coupled_file_list(FS)
    list_sos_rho = retrive_sos_sampling_file_list(FS)

    return list_pimc, list_jackknife, list_sos_vib, list_sos_rho


def retrive_file_paths_for_jackknife(FS):
    """return three lists of the full paths to each data file that might be used"""

    list_pimc = retrive_pimc_file_list(FS)
    list_sos_vib = retrive_sos_coupled_file_list(FS)
    list_sos_rho = retrive_sos_sampling_file_list(FS)

    return list_pimc, list_sos_vib, list_sos_rho


def extract_pimc_parameters(list_pimc, list_coupled, list_sampling):
    """make a list of all parameters whose dependencies are satisfied
    note that this function is tightly tied to the file name
    """

    # note that the way these splits are coded will pose problems if the naming scheme for sos is changed in the future
    value_dict = {"pimc_beads": 0, "basis_fxns": 0, "temperatures": 0}

    cL = map(lambda path: int(path.split("_B")[1].split(".json")[0]), list_coupled)
    sL = map(lambda path: int(path.split("_B")[1].split(".json")[0]), list_sampling)

    # parse file paths to find shared #'s' of basis functions
    value_dict["basis_fxns"] = list(set(cL) & set(sL))
    value_dict["basis_fxns"].sort()
    log.debug(value_dict["basis_fxns"])

    # parse file paths to find shared temperature values
    # for now we will leave this partially undeveloped
    tempL = map(lambda path: float(path.split("_T")[1].split("_X")[0]), list_pimc)
    # need to add thing here that checks temperatures inside sos file
    value_dict["temperatures"] = list(set(tempL))
    value_dict["temperatures"].sort()
    log.debug(value_dict["temperatures"])

    # parse file paths to find shared sample values
    xL = map(lambda path: int(path.split("_X")[1].split("_thermo")[0]), list_pimc)
    value_dict["samples"] = list(set(xL))
    value_dict["samples"].sort()
    log.debug(value_dict["samples"])

    # parse file paths to find pimc bead values
    pL = map(lambda path: int(path.split("_P")[1].split("_T")[0]), list_pimc)
    value_dict["pimc_beads"] = []
    for p in pL:
        if p not in value_dict["pimc_beads"]:
            value_dict["pimc_beads"].append(p)
    value_dict["pimc_beads"].sort()
    log.debug(value_dict["pimc_beads"])

    return value_dict


# I believe this function is decommissioned for the time being
def extract_jackknife_parameters(list_pimc, list_coupled, list_sampling):
    """make a list of all parameters whose dependencies are satisfied
    note that this function is tightly tied to the file name
    """

    # note that the way these splits are coded will pose problems if the naming scheme for sos is changed in the future
    value_dict = {"pimc_beads": 0, "basis_fxns": 0, "temperatures": 0}

    cL = map(lambda path: int(path.split("_B")[1].split(".json")[0]), list_coupled)
    sL = map(lambda path: int(path.split("_B")[1].split(".json")[0]), list_sampling)

    # parse file paths to find shared #'s' of basis functions
    value_dict["basis_fxns"] = list(set(cL) & set(sL))
    value_dict["basis_fxns"].sort()
    log.debug(value_dict["basis_fxns"])

    # parse file paths to find shared temperature values
    # for now we will leave this partially undeveloped
    tempL = map(lambda path: float(path.split("_T")[1].split("_J")[0]), list_pimc)
    # need to add thing here that checks temperatures inside sos file
    value_dict["temperatures"] = list(set(tempL))
    value_dict["temperatures"].sort()
    log.debug(value_dict["temperatures"])

    # # parse file paths to find shared sample values
    # xL = map(lambda path: int(path.split("_X")[1].split("_P")[0]), list_pimc)
    # value_dict["samples"] = list(set(xL))
    # value_dict["samples"].sort()
    # log.debug(value_dict["samples"])

    # parse file paths to find pimc bead values
    pL = map(lambda path: int(path.split("_P")[1].split("_T")[0]), list_pimc)
    value_dict["pimc_beads"] = []
    for p in pL:
        if p not in value_dict["pimc_beads"]:
            value_dict["pimc_beads"].append(p)
    value_dict["pimc_beads"].sort()
    log.debug(value_dict["pimc_beads"])

    return value_dict


def load_data(FS, P, B, T, pimcArgs, rhoArgs):
    """x"""

    path_data_points = FS.template_pimc.format(P=P, T=T, J="*")
    path_sos = FS.template_sos_rho.format(B=B)

    try:
        with np.load(path_data_points, allow_pickle=True) as npz_file:
            # TODO - should make a class function in minimal.BoxResult that returns the data
            # instead of using the specifiers "s_g", etc.
            # this reduces the number of places in the code that need to be changed
            pimcArgs["s_g"] = npz_file["s_g"]
            pimcArgs["s_rho"] = npz_file["s_rho"]
            # plus minus version
            if "s_gP" in npz_file:
                pimcArgs["s_gP"] = npz_file["s_gP"]
                pimcArgs["s_gM"] = npz_file["s_gM"]

        # TODO - should add stricter testing to make sure that the fraction itself isn't too small or big
        if np.any(pimcArgs["s_rho"] == 0.0):
            raise Exception("zeros in the denominator")

        with open(path_sos, "r") as rho_file:
            rho_dict = json.loads(rho_file.read())

        # TODO - should make a class function in a new module that handles sos stuff ??
        input_temp_index = rho_dict["temperature"].index(T)
        # make sure the temperature matches
        assert(T == rho_dict["temperature"][input_temp_index])

        rhoArgs["Z"] = rho_dict["Z_sampling"][input_temp_index]
        rhoArgs["E"] = rho_dict["E_sampling"][input_temp_index]
        rhoArgs["Cv"] = rho_dict["Cv_sampling"][input_temp_index]
        rhoArgs["alpha_plus"] = rhoArgs["Z"] / rho_dict["Z_sampling+beta"][input_temp_index]
        rhoArgs["alpha_minus"] = rhoArgs["Z"] / rho_dict["Z_sampling-beta"][input_temp_index]

    except OSError as err:
        # skip if we cannot obtain all the necessary data
        log.info("Skipped {:} and {:}".format(path_data_points, path_sos))
        return

    # We worked
    print(path_data_points)
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


if (__name__ == "__main__"):

    # catalogue available files
    pimcList, coupledList, samplingList = retrive_file_paths_for_jackknife()

    # find shared values
    argDict = extract_parameters(pimcList, coupledList, samplingList)

    # would be nice to have some feedback about what values there are and what are missing

    # manually select specific values from those available
    pimc_restriction = range(12, 101, 1) # at least 12 beads before we plot
    # pick the highest number of samples
    sample_restriction = argDict["samples"][-1]
    # pick the highest number of basis functions
    basis_restriction = argDict["basis_fxns"][-1]
    # temperature is currently fixed at 300K
    temperature_restriction = np.array([300])

    # intersect returns sorted, unique values that are in both of the input arrays
    argDict["temperatures"] = np.intersect1d(argDict["temperatures"], temperature_restriction)
    argDict["pimc_beads"] = np.intersect1d(argDict["pimc_beads"], pimc_restriction)
    argDict["basis_fxns"] = np.intersect1d(argDict["basis_fxns"], basis_restriction)
    argDict["samples"] = np.intersect1d(argDict["samples"], sample_restriction)

    # create a list of arguments for multiprocessing pool
    # arg_list = []
    # for X in argDict["samples"]:
    #     for P in argDict["pimc_beads"]:
    #         for T in argDict["temperatures"]:
    #             for B in argDict["basis_fxns"]:
    #                 arg_list.append((X, P, T, B))
    arg_list = [(X, P, T, B)
                for X in argDict["samples"]
                for P in argDict["pimc_beads"]
                for T in argDict["temperatures"]
                for B in argDict["basis_fxns"]
                ]

    arg_iterator = iter(arg_list)

    block_size = 10

    with mp.Pool(block_size) as p:
        p.starmap(main, arg_list)

    print("Finished")
