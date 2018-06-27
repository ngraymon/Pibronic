# stats.py

# system imports
import multiprocessing as mp
# import itertools as it
from functools import partial
# import collections
# import subprocess
# import socket
# import glob
import json
import sys
# import os

# third party imports
import numpy as np

# local imports
from . import jackknife as jk
from ..data import file_structure as fs
from ..data import file_name
from ..data import postprocessing as pp
from ..pimc import BoxResultPM
from .. import constants
from ..constants import boltzman


__all__ = [
           "calculate_alpha_terms",
           "add_harmonic_contribution",
           "estimate_basic_properties",
           "calculate_basic_property_terms",
           ]


__number_of_processes = 12


def calculate_basic_property_terms(*args):
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
    ratio_plus = g_plus / rho
    ratio_minus = g_minus / rho

    first_symmetric_derivative = np.mean(ratio_plus) * np.mean(alpha_plus)
    first_symmetric_derivative -= np.mean(ratio_minus) * np.mean(alpha_minus)
    first_symmetric_derivative /= (2. * delta_beta)  # constant factor

    second_symmetric_derivative = np.mean(ratio_plus) * np.mean(alpha_plus)
    second_symmetric_derivative -= 2. * np.mean(ratio)
    second_symmetric_derivative += np.mean(ratio_minus) * np.mean(alpha_minus)
    second_symmetric_derivative /= pow(delta_beta, 2)  # constant factor

    ret = [ratio,
           first_symmetric_derivative,
           second_symmetric_derivative,
           ratio_plus, alpha_plus, ratio_minus, alpha_minus,
           ]

    return ret


def basic_estimate_Z_monte_carlo(g_over_rho, number_of_samples):
    """ estimates the normalization of the quasi-probability disribution g(R) and its standard deviation """
    ""
    Z_MC = np.mean(g_over_rho)
    Z_err = np.std(g_over_rho, ddof=0)
    Z_err /= np.sqrt(number_of_samples - 1)
    return Z_MC, Z_err


def basic_estimate_internal_energy(first_symmetric_derivative, Z_MC):
    """ estimates the internal energy using finite difference , the error is always zero """
    E = -1. * np.mean(first_symmetric_derivative) / np.mean(Z_MC)
    E_err = 0.0  # can't be measured (in the basic case)
    return E, E_err


def basic_estimate_heat_capacity(second_symmetric_derivative, Z_MC, internal_energy, temperature):
    """ calculates the heat capacity using finite difference , the error is always zero """
    Cv = np.mean(second_symmetric_derivative) / np.mean(Z_MC)
    Cv -= pow(internal_energy, 2.)
    Cv /= boltzman * pow(temperature, 2.)  # 1 / kBT
    Cv_err = 0.0  # can't be measured
    return Cv, Cv_err


def estimate_basic_properties(*args):
    """ calculates the Z_MC, E, Cv, and their respective errors and returns a dictionary with 6 corresponding entries """
    X, T, g_r, sym1, sym2 = args

    Z_MC, Z_err = basic_estimate_Z_monte_carlo(g_r, X)
    E, E_err = basic_estimate_internal_energy(sym1, Z_MC)
    Cv, Cv_err = basic_estimate_heat_capacity(sym2, Z_MC, E, T)

    # easy to access storage
    # TODO - should we rename the key to Z_MC?
    return_dictionary = {"Z": Z_MC, "Z error": Z_err,
                         "E": E,    "E error": E_err,
                         "Cv": Cv,  "Cv error": Cv_err,
                         }
    return return_dictionary


def add_harmonic_contribution(input_dict, E_sampling, Cv_sampling):
    """ adds the constant harmonic contribution to the energy and the heat capacity """
    input_dict["E"] += E_sampling  # add the harmonic contribution to the energy
    input_dict["Cv"] += Cv_sampling  # add the harmonic contribution to the heat capacity
    return


def apply_parameter_restrictions(args):
    """ this is just a placeholder function for a possible idea, it doens't do anything right now """
    return  # don't do anything

    # manually select specific values from those available
    pimc_restriction = range(12, 101, 1)  # at least 12 beads before we plot
    # temperature is currently fixed at 300K
    temperature_restriction = np.array([300.00])

    # apply the restriction
    args["temperatures"] = np.intersect1d(args["temperatures"], temperature_restriction)
    args["pimc_beads"] = np.intersect1d(args["pimc_beads"], pimc_restriction)
    return


def starmap_wrapper(FS, P, T, statistical_operation):
    """ this function allows us to use the multiprocessing starmap in a convient way inside basic_statistical_analysis_of_pimc() and basic_jackknife_analysis_of_pimc()
    input is a FileStructure object, a bead value, a temperature value, and a function which preforms the calculation
    it loads all appropriate files
    it then calls the statistical_operation() function with these parameters
    finally it saves the returned dictionary to the appropriate *_thermo file
    """

    # create the empty data structs which we fill with data
    pimc_results = BoxResultPM()
    rhoData = {}

    # load the data
    pp.load_pimc_data(FS, P, T, pimc_results)
    pp.load_analytic_data(FS, T, rhoData)

    # preform the statistical analysis
    output_dict = statistical_operation(T, pimc_results, rhoData)

    # now we should add the hash identifiers to it
    output_dict["hash_vib"] = FS.hash_vib
    output_dict["hash_rho"] = FS.hash_rho

    # save the data to disk
    path = FS.template_jackknife.format(P=P, T=T, X=pimc_results.samples)

    with open(path, mode='w', encoding='UTF8') as target_file:
        target_file.write(json.dumps(output_dict))


def basic_statistical_analysis(temperature, pimc_result, analytic_data):
    """ takes a temperature, a BoxResult object type, and a dictionary of analytical data and calculates the basic statistical properties, Z, E, Cv and returns them in a dictionary"""

    # these names need to be cross referenced with the naming scheme in pimc.py
    data = [pimc_result.scaled_rho.view(),         # rho
            pimc_result.scaled_g.view(),           # g
            pimc_result.scaled_gofr_plus.view(),   # g+
            pimc_result.scaled_gofr_minus.view(),  # g-
            ]

    terms = calculate_basic_property_terms(constants.delta_beta, *data)
    basic_dict = estimate_basic_properties(pimc_result.samples, temperature, *terms)
    add_harmonic_contribution(basic_dict, analytic_data["E"], analytic_data["Cv"])

    return basic_dict


def alpha_statistical_analysis(temperature, pimc_result, analytic_data):
    """ takes a temperature, a BoxResult object type, and a dictionary of analytical data and calculates the basic statistical properties, Z, E, Cv and returns them in a dictionary"""

    # these names need to be cross referenced with the naming scheme in pimc.py
    data = [pimc_result.scaled_rho.view(),         # rho
            pimc_result.scaled_g.view(),           # g
            pimc_result.scaled_gofr_plus.view(),   # g+
            pimc_result.scaled_gofr_minus.view(),  # g-
            analytic_data["alpha_plus"],
            analytic_data["alpha_minus"],
            ]

    terms = calculate_alpha_terms(constants.delta_beta, *data)
    terms = terms[:-4]  # we don't need the last four things
    alpha_dict = estimate_basic_properties(pimc_result.samples, temperature, *terms)
    add_harmonic_contribution(alpha_dict, analytic_data["E"], analytic_data["Cv"])

    return alpha_dict


def statistical_analysis_of_pimc(path, id_data, id_rho=0, method="basic", location="local", samples=None):
    """ preform calculation of Z, E, Cv for the given model, using either basic or alpha/difference terms """
    FS = fs.FileStructure(path, id_data, id_rho)
    FS.generate_model_hashes()  # build the hashes so that we can check against them
    list_pimc = pp.retrive_pimc_file_list(FS)

    # a dictionary of lists
    args = {}
    args["pimc_beads"] = pp.extract_bead_paramater_list(list_pimc)
    args["temperatures"] = pp.extract_temperature_paramater_list(list_pimc)

    apply_parameter_restrictions(args)  # for now this does nothing

    # create a list of all the parameter combinations we need to analyze
    arg_list = [(FS, P, T)
                for P in args["pimc_beads"]
                for T in args["temperatures"]
                ]

    # choose what statistical analysis we are going to preform
    if method is "basic":
        operation = basic_statistical_analysis
    elif method is "alpha":
        operation = alpha_statistical_analysis

    # dispatch multiple processes to execute the analyze concurrently
    if location is "local":
        basic_wrapper = partial(starmap_wrapper, statistical_operation=operation)

        with mp.Pool(__number_of_processes) as p:
            p.starmap(basic_wrapper, arg_list)

    elif location is "server":
        assert False, "Need to write this code"

        # TODO - add a simple command to check that slurm is installed
        # os.system("sbatch --version")
        # the output should be something like "slurm*#.#.#""

        # TODO - write code that submits jobs to the server

    else:
        raise Exception(f"Invalid value for paramter location:({location:s})")

    return


# TODO - the current implementation will blindly overwrite each *_thermo file every time its run, which is problematic if we want both jackknife and normal results in the same output file, the simplest fix to this is to just write a function which does all the combined statistical actions
# TODO - on a second pass, this seems to be fixed by the need for jackknife to calculate the basic properties along the way, although this might be a potential issue? - worth looking into later


def basic_jackknife_analysis(temperature, pimc_result, analytic_data):
    """ takes a temperature, a BoxResult object type, and a dictionary of analytical data and calculates the basic statistical properties, Z, E, Cv and returns them in a dictionary"""

    # these names need to be cross referenced with the naming scheme in pimc.py
    data = [pimc_result.scaled_rho.view(),         # rho
            pimc_result.scaled_g.view(),           # g
            pimc_result.scaled_gofr_plus.view(),   # g+
            pimc_result.scaled_gofr_minus.view(),  # g-
            ]

    T = temperature
    X = pimc_result.samples
    dB = constants.delta_beta

    terms = calculate_basic_property_terms(dB, *data)
    jk_terms = jk.calculate_jackknife_terms(X, terms)

    basic_dict = estimate_basic_properties(X, T, *terms)
    jk_dict = jk.estimate_jackknife(X, T, dB, basic_dict, *jk_terms)

    add_harmonic_contribution(basic_dict, analytic_data["E"], analytic_data["Cv"])
    add_harmonic_contribution(jk_dict, analytic_data["E"], analytic_data["Cv"])

    # create the output dictionary and rename the jackknife terms
    output_dict = basic_dict.copy()  # does this need to be a deep copy?
    for key in jk_dict.keys():
        output_dict["jk_" + key] = jk_dict[key]

    return output_dict


def alpha_jackknife_analysis(temperature, pimc_result, analytic_data):
    """ takes a temperature, a BoxResult object type, and a dictionary of analytical data and calculates the basic statistical properties, Z, E, Cv and returns them in a dictionary"""

    # these names need to be cross referenced with the naming scheme in pimc.py
    data = [pimc_result.scaled_rho.view(),         # rho
            pimc_result.scaled_g.view(),           # g
            pimc_result.scaled_gofr_plus.view(),   # g+
            pimc_result.scaled_gofr_minus.view(),  # g-
            analytic_data["alpha_plus"],
            analytic_data["alpha_minus"],
            ]

    T = temperature
    X = pimc_result.samples
    dB = constants.delta_beta

    terms = calculate_alpha_terms(dB, *data)
    # we don't need the 1sym or 2sym for the jackknife terms
    jk_terms = jk.calculate_alpha_jackknife_terms(X, dB, terms[0], *terms[3:7])
    terms = terms[:-4]  # we don't need the last four things
    alpha_dict = estimate_basic_properties(X, T, *terms)
    jk_dict = jk.estimate_jackknife(X, T, dB, alpha_dict, *jk_terms)

    add_harmonic_contribution(alpha_dict, analytic_data["E"], analytic_data["Cv"])
    add_harmonic_contribution(jk_dict, analytic_data["E"], analytic_data["Cv"])

    # create the output dictionary and rename the jackknife terms
    output_dict = alpha_dict.copy()  # does this need to be a deep copy?
    for key in jk_dict.keys():
        output_dict["jk_" + key] = jk_dict[key]

    return output_dict


def jackknife_analysis_of_pimc(path, id_data, id_rho=0, method="basic", location="local", samples=None):
    """ preform calculation of Z, E, Cv for the given model, using either basic or alpha/difference terms with the jackknife method"""
    FS = fs.FileStructure(path, id_data, id_rho)
    FS.generate_model_hashes()  # build the hashes so that we can check against them
    list_pimc = pp.retrive_pimc_file_list(FS)

    # a dictionary of lists
    args = {}
    args["pimc_beads"] = pp.extract_bead_paramater_list(list_pimc)
    args["temperatures"] = pp.extract_temperature_paramater_list(list_pimc)

    apply_parameter_restrictions(args)  # for now this does nothing

    # create a list of all the parameter combinations we need to analyze
    arg_list = [(FS, P, T)
                for P in args["pimc_beads"]
                for T in args["temperatures"]
                ]
    # choose what statistical analysis we are going to preform
    if method is "basic":
        operation = basic_jackknife_analysis
    elif method is "alpha":
        operation = alpha_jackknife_analysis

    # dispatch multiple processes to execute the analyze concurrently
    if location is "local":
        jackknife_wrapper = partial(starmap_wrapper, statistical_operation=operation)
        with mp.Pool(__number_of_processes) as p:
            p.starmap(jackknife_wrapper, arg_list)

    elif location is "server":
        assert False, "Need to write this code"

        # TODO - add a simple command to check that slurm is installed
        # os.system("sbatch --version")
        # the output should be something like "slurm*#.#.#""

        # TODO - write code that submits jobs to the server

    else:
        raise Exception(f"Invalid value for paramter location:({location:s})")

    return


def testing_execution(path, id_data, id_rho):
    """ temporary """
    statistical_analysis_of_pimc(path, id_data, id_rho, method="basic")
    statistical_analysis_of_pimc(path, id_data, id_rho, method="alpha")
    jackknife_analysis_of_pimc(path, id_data, id_rho, method="basic")
    jackknife_analysis_of_pimc(path, id_data, id_rho, method="alpha")
    return


if (__name__ == "__main__"):
    """ this is only used for testing at the moment """
    test_path = "/work/ngraymon/pimc/testing/"
    id_data = int(sys.argv[1])
    id_rho = 0
    testing_execution(test_path, id_data, id_rho)

    print("Finished")
