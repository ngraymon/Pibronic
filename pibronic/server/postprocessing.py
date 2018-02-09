# postprocessing.py

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
from ..data import file_structure
# from ..server import job_boss

path_root = "/work/ngraymon/pimc/"

file_suffix = (
                "D{id_data:d}_"
                "R{id_rho:d}_"
                "P{number_of_beads:d}_"
                "T{temperature:.2f}"
                )
sos_suffix = "sos_B{basis_size:d}.json"

def retrive_pimc_file_list(files):
    """find a list of all data files that might used"""

    # retrive pimc files
    globPath = files.path_rho_results + "D{:d}_*_thermo".format(files.id_data)
    list_pimc = [file for file in glob.glob(globPath)]

    # retrive coupled files
    globPath = files.path_vib_params + "sos_B*.json"
    list_sos_vib = [file for file in glob.glob(globPath)]

    # retrive sampling files
    globPath = files.path_rho_params + "sos_B*.json"
    list_sos_rho = [file for file in glob.glob(globPath)]

    return list_pimc, list_sos_vib, list_sos_rho


def retrive_jackknife_file_list(files):
    """find a list of all data files that might used"""

    # retrive pimc files
    globPath = files.path_rho_results + "D{:d}_*_data_points.npz".format(files.id_data)
    list_pimc = [file for file in glob.glob(globPath)]

    # retrive coupled files
    globPath = files.path_vib_params  + "sos_B*.json"
    list_sos_vib = [file for file in glob.glob(globPath)]

    # retrive sampling files
    globPath = files.path_rho_params + "sos_B*.json"
    list_sos_rho = [file for file in glob.glob(globPath)]

    return list_pimc, list_sos_vib, list_sos_rho


def extract_pimc_parameters(list_pimc, list_coupled, list_sampling):
    """make a list of all parameters whose dependencies are satisfied"""

    # note that the way these splits are coded
    # will pose problems if the naming scheme for sos is changed in the future
    value_dict = {"pimc_beads":0, "basis_fxns":0, "temperatures":0}

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



def extract_jackknife_parameters(list_pimc, list_coupled, list_sampling):
    """make a list of all parameters whose dependencies are satisfied"""

    # note that the way these splits are coded
    # will pose problems if the naming scheme for sos is changed in the future
    value_dict = {"pimc_beads":0, "basis_fxns":0, "temperatures":0}

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


def load_data(P, B, T, path_root, id_data, id_rho, pimcArgs, rhoArgs):
    """"""
    path_root += "data_set_{:d}/rho_{:d}/".format(id_data, id_rho)
    path_results = path_root + "results/"
    path_params = path_root + "parameters/"

    path_data_points = path_results + file_suffix
    path_data_points = path_data_points.format( id_data=id_data,
                                                id_rho=id_rho,
                                                number_of_beads=P,
                                                temperature=T,
                                                )

    path_sos = path_params + sos_suffix.format(basis_size=B)

    try:
        with np.load(path_data_points, allow_pickle=True) as npz_file:
            pimc_data["s_g"] = npz_file["s_g"]
            pimc_data["s_rho"] = npz_file["s_rho"]
            # plus minus version
            if "s_gP" in npz_file:
                pimc_data["s_gP"] = npz_file["s_gP"]
                pimc_data["s_gM"] = npz_file["s_gM"]


        if np.any(pimcArgs["s_rho"] == 0.0):
            raise Exception("zeros in the denominator")

        with open(path_sos, "r") as rho_file:
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
    pimcList, coupledList, samplingList = retrive_file_list()

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
    arg_list = [(X, P, T, B)                    \
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