"""
Handles all file processing after calculations are run

Provides functions for collating files for statistical analaysis (such as jackknife) and plotting
"""

# system imports
# import multiprocessing as mp
# import itertools as it
import collections
# import subprocess
# import socket
import json
import glob
# import sys
import os
from os.path import join

# third party imports
import numpy as np
# from numpy import newaxis as NEW
# from numpy import float64 as F64

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


# note that the way the split()'s are coded for all the following extract_ functions
# will pose problems if the naming scheme for the files is changed
# it might be good to eventually link them to the file_name module in the future

def extract_bead_value_from_trotter_file_path(file_path):
    """ does what it says, returns an int """
    return int(file_path.split("trotter_P")[1].split("_B"))


def extract_trotter_paramater_list(list_trotter):
    """ takes a list of file-paths to results from trotter calculations and return a dictionary with integer keys representing possible basis sizes, whose corresponding values are lists of all possible bead values for that given basis size. """
    tL = map(extract_bead_value_from_trotter_file_path, list_trotter)

    trotterDict = collections.defaultdict(list)
    for path in tL:
        trotterDict[int(path[1].split(".json")[0])].append(int(path[0]))

    return trotterDict


def extract_basis_value_from_sos_file_path(file_path):
    """ does what it says, returns an int """
    return int(file_path.split("_B")[1].split(".json")[0])


def extract_sos_basis_paramater_list(list_vib, list_rho):
    """ takes a list of file-paths to results from sos calculations (both coupled and rho) and return a list of all the unique """
    cL = map(extract_basis_value_from_sos_file_path, list_vib)
    sL = map(extract_basis_value_from_sos_file_path, list_rho)

    list_sos = list(set(cL) & set(sL))
    list_sos.sort()
    return list_sos


def extract_bead_value_from_pimc_file_path(file_path):
    """ does what it says, returns an int """
    return int(file_path.split("P")[1].split("_T")[0])


def extract_bead_paramater_list(list_pimc):
    """ takes a list of file-paths to results from pimc calculations and return a list of all the unique bead values"""
    pL = map(extract_bead_value_from_pimc_file_path, list_pimc)
    list_bead = list(set(pL))  # the use of the set object removes all duplicate elements
    list_bead.sort()
    return list_bead


def extract_temperature_value_from_pimc_file_path(file_path):
    """ does what it says, returns a float"""
    return float(file_path.split("_T")[1].split("_J")[0])


def extract_temperature_paramater_list(list_pimc):
    """ takes a list of file-paths to results from pimc calculations and return a list of all the unique temperature values (as floats)"""

    # this option would be to specifically only select temperatures from thermo files instead of just generally from all output files, it is not clear which is better
    # tempL = map(lambda path: int(path.split("_T")[1].split("_thermo")[0]), list_pimc)
    tempL = map(extract_temperature_value_from_pimc_file_path, list_pimc)
    list_temperature = list(set(tempL))  # the use of the set object removes all duplicate elements
    list_temperature.sort()
    return list_temperature


def extract_job_value_from_pimc_file_path(file_path):
    """ does what it says, returns an int"""
    return int(file_path.split("_J")[1].split("_data_")[0])


def extract_parameter_lists(list_pimc, list_vib, list_rho):
    """ just assume that we directly use the extract_trotter_paramater_list() function for now """
    bL = extract_sos_basis_paramater_list(list_vib, list_rho)
    pL = extract_bead_paramater_list(list_pimc)
    tL = extract_temperature_paramater_list(list_pimc)
    return pL, tL, bL


def extract_bead_value_from_thermo_file_path(file_path):
    """ does what it says, returns an int """
    return int(file_path.split("P")[1].split("_T")[0])


def extract_temperature_value_from_thermo_file_path(file_path):
    """ does what it says, returns a float"""
    return float(file_path.split("_T")[1].split("_X")[0])


def extract_sample_value_from_thermo_file_path(file_path):
    """ does what it says, returns an int"""
    return int(file_path.split("_X")[1].split("_thermo")[0])


def prune_results_using_hashes(FS, list_pimc):
    """ takes a list of file paths (strings) to different results and a FileStructure object
    returns a subset of the input list where each returned file path exists and has a
    'valid' hash, i.e. the same as in the FileStructure object"""
    output_list = []
    for file_path in list_pimc:
        if file_path[-4:] == '.npz':
            with np.load(file_path, mmap_mode='r') as file:
                if ('hash_vib' not in file) or ('hash_rho' not in file):
                    continue
                if file['hash_vib'] == FS.hash_vib and file['hash_rho'] == FS.hash_rho:
                    output_list.append(file_path)
        elif file_path[-6:] == 'thermo':
            with open(file_path, 'r') as file:
                data = json.loads(file.read())
            if ('hash_vib' not in data) or ('hash_rho' not in data):
                continue
            if data['hash_vib'] == FS.hash_vib and data['hash_rho'] == FS.hash_rho:
                output_list.append(file_path)
        else:
            raise Exception("file path {:s} undefined".format(file_path))
    return output_list


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


def load_pimc_data(FS, P, T, pimc_results):
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

    # TODO - add support for passing in the optional desired_number_of_samples paramter to the load_multiple_results method
    return


def load_analytic_data(FS, T, analytic):
    """ load data from analytic_results.txt with same T into the dictionary analytic"""

    path = FS.path_analytic_rho

    try:
        assert os.path.isfile(path), f"This file doesn't exist:\n{path:s}"
        with open(path, "r") as file:
            in_dict = json.loads(file.read())

            # make sure the analytic data is up to date!!
            assert in_dict["hash_vib"] == FS.hash_vib, "wrong vib hash"
            assert in_dict["hash_rho"] == FS.hash_rho, "wrong rho hash"
            # TODO - should make a class function in a new module that handles analytic stuff ??
            temperature = str(T)
            assert temperature in in_dict.keys(), "no analytical results for temperature {:s} in file {:s}".format(temperature, path)

            analytic["Z"] = in_dict[temperature]["Z_sampling"]
            analytic["E"] = in_dict[temperature]["E_sampling"]
            analytic["Cv"] = in_dict[temperature]["Cv_sampling"]

            analytic["alpha_plus"] = analytic["Z"] / in_dict[temperature]["Z_sampling+beta"]
            analytic["alpha_minus"] = analytic["Z"] / in_dict[temperature]["Z_sampling-beta"]

    except OSError as err:
        # skip if we cannot obtain all the necessary data
        print("Skipped data from {:s} at temperature {:.2f}".format(path, T))
        raise err  # we cannot proceed at the moment since stats.py needs this file
        return

    return


def load_rho_sos_data(FS, P, B, T, rho_args):
    """ load data from all files with same P and T into the dictionary rho_args"""

    path = FS.template_sos_rho.format(B=B)

    try:
        assert os.path.isfile(path), f"This file doesn't exist:\n{path:s}"
        with open(path, "r") as file:
            rho_dict = json.loads(file.read())
            # TODO - should make a class function in a new module that handles sos stuff ??
            input_temp_index = rho_dict["temperature"].index(T)
            # make sure the temperature matches
            assert T == rho_dict["temperature"][input_temp_index], "different temperatures"

            rho_args["Z"] = rho_dict["Z_sampling"][input_temp_index]
            rho_args["E"] = rho_dict["E_sampling"][input_temp_index]
            rho_args["Cv"] = rho_dict["Cv_sampling"][input_temp_index]

            rho_args["alpha_plus"] = rho_args["Z"] / rho_dict["Z_sampling+beta"][input_temp_index]
            rho_args["alpha_minus"] = rho_args["Z"] / rho_dict["Z_sampling-beta"][input_temp_index]

    except OSError as err:
        # skip if we cannot obtain all the necessary data
        print("Skipped {:s}".format(path))
        return

    return
