#!python3

# system imports
import subprocess
import argparse
import json
import sys
import os
from os.path import join, isfile, dirname, realpath

# third party imports

# local imports
from pibronic import constants
from pibronic.log_conf import log
from pibronic.data import file_structure as fs
from pibronic.server.job_boss import subprocess_run_wrapper

""" this dictionary is used to map the output of processes running Julia scripts
each Julia script prints out some number of "Name: value" strings
we want to capture the value's in those strings
however we may want to associate them with different Names
the keys of this dictionary are the new Names we wish to use, and the values are the old Names
"""
keyDict = {
    "Z_MC": "Z_MC",
    "E_MC": "E_MC",
    "Cv_MC": "Cv_MC",
    "SvN_MC": "SvN_MC",
    "S2_MC": "S2_MC",
    "Z_trotter": "Z_trotter",
    "E_trotter": "E_trotter",
    "Cv_trotter": "Cv_trotter",
    "SvN_trotter": "SvN_trotter",
    "S2_trotter": "S2_trotter",
    "Z_sos": "Z_sos",
    "E_sos": "E_sos",
    "Cv_sos": "Cv_sos",
    "SvN_sos": "SvN_sos",
    "S2_sos": "S2_sos",
    "E0_exact": "E0_exact",
    "SvN_exact": "SvN_exact",
    "S2_exact": "S2_exact",
    "Z_rho": "Z_analytical",
    "E_rho": "E_analytical",
    "Cv_rho": "Cv_analytical",
    "SvN_rho": "SvN_analytical",
    "S2_rho": "S2_analytical",
    # "Z_rho+beta": "Zrho+(beta)",
    # "Z_rho-beta": "Zrho-(beta)",
    "beta": "beta",
    "tau": "tau",
}


def parse_ouput(byte_string):
    """ takes the byte string output from the Julia script as input
    splits it into a list of string representations of each line
    makes a dictionary by splitting each line at the semicolon
    with the key as all characters on the left and the value as all characters on the right
    """
    rawOutput = ''.join(list(byte_string.decode()))
    lineList = [string for string in rawOutput.split("\n") if string is not '']
    # quick check to make sure that our assumption about the structure of the output is correct
    for line in lineList:
        assert ":" in line
    # the following step requires that every line has a semicolon
    output_dict = dict(line.split(":") for line in lineList)
    return output_dict


def convert_keys(old_dict, output_dict, input_beta):
    """ uses the keyDict to map keys from the output_dict to new keys
    stores the resulting key,value pairs in a new_dict
    calls update on old_dict using this new_dict it has created
    """
    temp_dict = {}

    for newKey, oldKey in keyDict.items():
        if oldKey in output_dict:
            temp_dict[newKey] = float(output_dict[oldKey])

    key_T = "{:.2f}".format(constants.extract_T_from_beta(input_beta))

    new_dict = {key_T: temp_dict}

    old_dict.update(new_dict)
    # print(old_dict)
    return


def compute(command, path_src, old_dict, input_beta):
    """ x """
    log.flow("Computing")
    log.flow(command)

    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (out, error) = p.communicate()
    print(out, error)

    # check to make sure the job wasn't killed
    if "Killed" in error.decode():
        print(f"We ran into an error: {error.decode():s}")
        return False

    output_dict = parse_ouput(out)
    # print(output_dict)

    # check to make sure that we didn't get 0 results
    if not bool(output_dict):
        print(f"The results dictionary after running Julia code on {path_src} is empty, therefore the calculation did not complete successfully")
        return False

    convert_keys(old_dict, output_dict, input_beta)

    return True


def validate_old_rho_data(old_dict, FS):
    """ check if the old hashes match the new ones,
    otherwise we have to throw away all the old data
    """
    if "hash_vib" not in old_dict or "hash_rho" not in old_dict:
        # throw away all the old data
        old_dict.clear()

    elif old_dict["hash_vib"] != FS.hash_vib or old_dict["hash_rho"] != FS.hash_rho:
        # throw away all the old data
        old_dict.clear()

    # add the new hashes
    old_dict["hash_vib"] = FS.hash_vib
    old_dict["hash_rho"] = FS.hash_rho
    return


def validate_old_model_data(old_dict, FS):
    """ check if the old hashes match the new ones,
    otherwise we have to throw away all the old data
    """
    if "hash_vib" not in old_dict:
        # throw away all the old data
        old_dict.clear()

    elif old_dict["hash_vib"] != FS.hash_vib:
        # throw away all the old data
        old_dict.clear()

    # add the new hashes
    old_dict["hash_vib"] = FS.hash_vib
    return


def base_func(FS, beta, path_dst=None, path_src=None, validate=None, command=None):

    old_dict = {}

    if isfile(path_dst):
        with open(path_dst, 'r') as file:
            data = file.read()
            if len(data) > 1:
                old_dict = json.loads(data)

    validate(old_dict, FS)

    if compute(command, path_src, old_dict, beta):
        validate(old_dict, FS)
        with open(path_dst, 'w') as file:
            json.dump(old_dict, file)
    else:
        raise Exception("Calculation did not complete successfully")

    return


def analytic_wrapper(FS, beta, **kwargs):
    """ this wrapper evaluates the given command with the appropriate arguments
    before passing the args to the base_func()
    """
    kwargs["command"] = kwargs["command"].format(F=kwargs["path_src"], T=beta)
    base_func(FS, beta, **kwargs)
    return


def analytic_of_sampling_model(FS, beta):
    """ attempts to analytically calculate Z from the sampling model (rho_#)
    checks for previous results data, reads it in, then tosses the data if the hashes are not valid
    attempts to execute the analytical Julia script, and if successful saves the output
    writes the combined old data and new data to the appropriate file
    """
    kwargs = {}
    kwargs["command"] = construct_command_dictionary()["analytical uncoupled"]
    kwargs["validate"] = validate_old_rho_data
    kwargs["path_dst"] = FS.path_analytic_rho
    kwargs["path_src"] = FS.path_rho_model
    analytic_wrapper(FS, beta, **kwargs)
    return


def analytic_of_original_coupled_model(FS, beta):
    """ attempts to analytically calculate Z from the original model
    checks for previous results data, reads it in, then tosses the data if the hashes are not valid
    attempts to execute the analytical Julia script, and if successful saves the output
    writes the combined old data and new data to the appropriate file
    """
    kwargs = {}
    kwargs["command"] = construct_command_dictionary()["analytical coupled"]
    kwargs["validate"] = validate_old_model_data
    kwargs["path_dst"] = FS.path_analytic_orig
    kwargs["path_src"] = FS.path_orig_model
    analytic_wrapper(FS, beta, **kwargs)
    return


def sos_wrapper(FS, basis_size, beta, **kwargs):
    """ this wrapper evaluates the given command with the appropriate arguments
    before passing the args to the base_func()
    """
    kwargs["command"] = kwargs["command"].format(F=FS.path_vib_model, T=beta, B=basis_size)
    base_func(FS, beta, **kwargs)
    return


def sos_of_coupled_model(FS, basis_size, beta):
    """ attempts to calculate Z using SOS
    checks for previous results data, reads it in, then tosses the data if the hashes are not valid
    attempts to execute the SOS Julia script, and if successful saves the output
    writes the combined old data and new data to the appropriate file
    """
    kwargs = {}
    kwargs["command"] = construct_command_dictionary()["sos"]
    kwargs["validate"] = validate_old_model_data
    kwargs["path_dst"] = FS.template_sos_vib.format(B=basis_size)
    kwargs["path_src"] = kwargs["path_dst"]
    sos_wrapper(FS, basis_size, beta, **kwargs)
    return


def trotter_wrapper(FS, nbeads, basis_size, beta, **kwargs):
    """ this wrapper evaluates the given command with the appropriate arguments
    before passing the args to the base_func()
    """
    kwargs["command"] = kwargs["command"].format(F=FS.path_vib_model, T=beta,
                                                 B=basis_size, P=nbeads)
    base_func(FS, beta, **kwargs)
    return


def trotter_of_coupled_model(FS, nbeads, basis_size, beta):
    """ attempts to calculate Z using SOS, including the trotter error
    checks for previous results data, reads it in, then tosses the data if the hashes are not valid
    attempts to execute the trotter Julia script, and if successful saves the output
    writes the combined old data and new data to the appropriate file
    """
    kwargs = {}
    kwargs["command"] = construct_command_dictionary()["trotter"]
    kwargs["validate"] = validate_old_model_data
    kwargs["path_dst"] = FS.template_trotter_vib.format(P=nbeads, B=basis_size)
    kwargs["path_src"] = kwargs["path_dst"]
    trotter_wrapper(FS, nbeads, basis_size, beta, **kwargs)
    return


def iterate_method(FS, n_iterations=50):
    """ this is just a wrapper for the iterative method at the moment
    it doesn't check for old data - it just regenerates the output every time
    """
    path_iterate = FS.path_iter_model

    command = construct_command_dictionary()["iterative"]
    command = command.format(F=FS.path_vib_model, P=path_iterate,
                             VS=FS.path_iter_mat, N=n_iterations,
                             )
    # print("command", command)

    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (out, error) = p.communicate()
    # print(out.decode())
    # print(error.decode())

    # we assume there were no issues!
    # TODO - this should be fixed after the iterative method has been finalized
    return


def prepare_julia(version="1.0.0"):
    """preform any tasks necessary to setup the environment for executing Julia code"""

    # a dynamic approach to checking possibly Julia versions would be preferred
    assert version in ["0.7.0", "1.0.0", "1.0.1"], "Incompatible Julia version"
    result = subprocess_run_wrapper(['julia', '-v'])

    if not ("julia version" in result.stdout):
        cmd = ['modulecmd', 'python', 'load', f'Julia/{version:s}']
        result = subprocess_run_wrapper(cmd)
        exec(result.stdout)  # this line is what ACTUALLY executes the module load
    return


def construct_command_dictionary():
    """x"""

    prepare_julia()
    cmd = ['julia', '-e', 'import VibronicToolkit; print(pathof(VibronicToolkit))']
    result = subprocess_run_wrapper(cmd)
    assert result.stderr == '', f"{result.stderr} happened!"

    path_root_jl_file = realpath(result.stdout)  # get the path to */src/*.jl
    path_root_folder = dirname(dirname(path_root_jl_file))  # go up 2 directories
    path_bin = join(path_root_folder, "pibronic_bin" + os.sep)  # the path to */bin/

    j_path = "julia " + path_bin

    d = {
         "analytical coupled":
         " ".join([j_path + "analytical.jl",
                   "--conf {F:}",
                   "--beta {T:}",
                   "--uncoupled",
                   ]),
         "analytical uncoupled":
         " ".join([j_path + "analytical.jl",
                   "--conf {F:}",
                   "--beta {T:}",
                   ]),
         "sos":
         " ".join([j_path + "sos.jl",
                   "--conf {F:}",
                   "--beta {T:}",
                   "--basis-size {B:}",
                   ]),
         "trotter":
         " ".join([j_path + "trotter.jl",
                  "--conf {F:}",
                   "--beta {T:}",
                   "--basis-size {B:}",
                   "--num-links {P:}",
                   ]),
         # if --sampling-conf is not provided the diagonal of the Hamiltonian is used by default
         "sampling":
         " ".join([j_path + "sampling.jl",
                   "--conf {F:}",
                   "--beta {T:}",
                   "--num-links {P:}",
                   "--num-samples {X:}",
                   # "--sampling-conf {:}",
                   ]),
         "finite difference sampling":
         " ".join([j_path + "sampling.jl",
                   "--conf {F:}",
                   "--beta {T:}",
                   "--num-links {P:}",
                   "--num-samples {X:}",
                   "--dbeta {:}",
                   "--sampling-beta {:}",
                   "--samplings-conf {:}",
                   ]),
         "iterative":
         " ".join([j_path + "iterative_decomposition.jl",
                  "--conf {F:}",
                   "--max-iter {N:}",
                   "--out-conf {P:}",
                   "--out-vs {VS:}",
                   ]),
         }

    return d


def main(**kwargs):
    """main function for testing or using from the command line"""

    # parse input, and set any default values
    path = kwargs.get('path')
    id_data = kwargs.get('id_data', 11)
    id_rho = kwargs.get('id_rho', 0)
    temperature = kwargs.get('temperature', 300.00)
    beta = constants.beta(temperature)
    nbeads = kwargs.get('beads', 10)
    basis_size = kwargs.get('basis_size', 2)
    method = kwargs.get('method', 'iterate')

    # generate the file structure object
    FS = fs.FileStructure(path, id_data, id_rho)
    FS.generate_model_hashes()

    # execute the specified (or default) method
    if method == "iterate":
        iterate_method(FS)
    elif method == "analytical_original":
        analytic_of_original_coupled_model(FS, beta)
    elif method == "analytical_sampling":
        analytic_of_sampling_model(FS, beta)
    elif method == "trotter_coupled":
        trotter_of_coupled_model(FS, nbeads=nbeads, basis_size=basis_size, beta=beta)
    elif method == "sos":
        sos_of_coupled_model(FS, basis_size=basis_size, beta=beta)
    else:
        raise Exception(f"The provided method {method} has not been added to the \
                         if/elif statement in the main() function of julia_wrapper.py")


if (__name__ == "__main__"):
    """call from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str,
                        help="Path to the root directory where the calculation will occur")
    parser.add_argument('-t', '--temperature', type=float, default=300.00,
                        help="Temperature in Kelvin")
    parser.add_argument('--beta', type=float, help="Beta value")
    parser.add_argument('-d', '--id_data', type=int, required=True, help="Data set id number")
    parser.add_argument('-r', '--id_rho', type=int, default=0, help="Rho id number")
    parser.add_argument('-p', '--nbeads', type=int,
                        help="Number of beads if calling a sampling method")
    parser.add_argument('-B', '--basis_size', type=int,
                        help="Number of basis functions if calling a trotter/sos method")
    parser.add_argument('-m', '--method', type=str, help="Specify a certain method")
    args = parser.parse_args()

    if not args.id_rho >= 0:
        parser.error("rho id must be >= 0")

    if args.temperature and args.beta:
        parser.error("Cannot provide both a temperature and a beta value")

    if args.temperature and not args.temperature > 0.0:
        parser.error("Temperature must be > 0.0")

    if args.nbeads:
        if args.method and 'sampling' not in args.method:
            parser.error("Number of beads should only be specified if using a sampling method")
        if not args.nbeads >= 3:
            parser.error("Number of beads must be >= 3")

    if args.basis_size:
        if args.method and ('trotter' not in args.method and 'sos' not in args.method):
            parser.error("Basis size should only be specified if using a trotter/sos method")
        if not args.basis_size >= 2:
            parser.error("Basis size must be >= 2")

    if args.method:
        cmd_dict = construct_command_dictionary()
        if args.method not in cmd_dict:
            valid_methods = list(cmd_dict.keys())
            s = f"Provided method {args.method} is not one of the valid methods:\n{valid_methods}"
            parser.error(s)

    kwargs = vars(args)
    main(**kwargs)
