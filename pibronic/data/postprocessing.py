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
# from .. import constants
# from ..constants import hbar
from ..log_conf import log
from ..data import file_name
# from ..data import file_structure
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


# I believe this function is decommissioned for the time being
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
    pL = map(lambda path: int(path.split("/P")[1].split("_T")[0]), list_pimc)
    value_dict["pimc_beads"] = []
    for p in pL:
        if p not in value_dict["pimc_beads"]:
            value_dict["pimc_beads"].append(p)
    value_dict["pimc_beads"].sort()
    log.debug(value_dict["pimc_beads"])

    return value_dict


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
    pL = map(lambda path: int(path.split("results/P")[1].split("_T")[0]), list_pimc)
    value_dict["pimc_beads"] = []
    for p in pL:
        if p not in value_dict["pimc_beads"]:
            value_dict["pimc_beads"].append(p)
    value_dict["pimc_beads"].sort()
    log.debug(value_dict["pimc_beads"])

    return value_dict


def load_data(FS, P, B, T, pimc_results, rhoArgs):
    """ load data from all files with same P and T"""

    """
    this usuage of FS.template_pimc.format(P=P, T=T, J="*") raises a good question about the design!
    possible ways to implement:
        - some combination of partial
        - write function for each template_* member of file_structure to replace .format()
        - write function for file_name
    currently solved by:
        self.template_pimc = self.path_rho_results + file_name.pimc(J="{J:s}")
    """
    path_data_points = FS.template_pimc.format(P=P, T=T, J="*")
    list_of_files = [file for file in glob.glob(path_data_points)]
    pimc_results.load_multiple_results(list_of_files)
    path_sos = FS.template_sos_rho.format(B=B)

    try:
        with open(path_sos, "r") as rho_file:
            rho_dict = json.loads(rho_file.read())
            # TODO - should make a class function in a new module that handles sos stuff ??
            input_temp_index = rho_dict["temperature"].index(T)
            # make sure the temperature matches
            assert T == rho_dict["temperature"][input_temp_index], "different temperatures"

            rhoArgs["Z"] = rho_dict["Z_sampling"][input_temp_index]
            rhoArgs["E"] = rho_dict["E_sampling"][input_temp_index]
            rhoArgs["Cv"] = rho_dict["Cv_sampling"][input_temp_index]

            rhoArgs["alpha_plus"] = rhoArgs["Z"] / rho_dict["Z_sampling+beta"][input_temp_index]
            rhoArgs["alpha_minus"] = rhoArgs["Z"] / rho_dict["Z_sampling-beta"][input_temp_index]

    except OSError as err:
        # skip if we cannot obtain all the necessary data
        print("Skipped {:} and {:}".format(path_data_points, path_sos))
        return

    return
