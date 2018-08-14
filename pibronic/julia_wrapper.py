#!python3

# system imports
import subprocess
import json
import sys
import os
from os.path import join, isfile

# third party imports

# local imports
from pibronic import constants
from pibronic.log_conf import log
from pibronic.vibronic import vIO
from pibronic.data import file_structure as fs


def compute(command, path_src, old_dict, input_beta):
    """x"""
    log.flow("Computing")
    log.flow(command)

    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (out, error) = p.communicate()

    # check to make sure the job wasn't killed
    if "Killed" in error.decode():
        print(f"We ran into an error: {error.decode():s}")
        return False

    rawOutput = ''.join(list(out.decode()))
    lineList = [string for string in rawOutput.split("\n") if string is not '']
    output_dict = dict(line.split(":") for line in lineList)

    print(output_dict)
    # check to make sure that we didn't get 0 results
    if not bool(output_dict):
        print(f"The results dictionary after running julia code on {path_src} is empty, therefore the calculation did not complete successfully")
        return False

    temp_dict = {}
    keyDict = {
        "Z_coupled": "ZH",
        "Z_harmonic": "Zrho",
        "Z_sampling": "Zrho",
        "Z_sampling+beta": "Zrho+(beta)",
        "Z_sampling-beta": "Zrho-(beta)",
        "E_coupled": "E",
        "E_harmonic": "Erho",
        "E_sampling": "Erho",
        "Cv_coupled": "Cv",
        "Cv_harmonic": "Cvrho",
        "Cv_sampling": "Cvrho",
        #
        "beta": "beta",
        "tau": "tau",
    }

    for newKey, oldKey in keyDict.items():
        if oldKey in output_dict:
            temp_dict[newKey] = float(output_dict[oldKey])

    key_T = "{:.2f}".format(constants.extract_T_from_beta(input_beta))

    new_dict = {key_T: temp_dict}

    old_dict.update(new_dict)
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

# TODO - it looks like we can refactor analytic_of_sampling_model()
# and analytic_of_original_coupled_model()


def analytic_of_sampling_model(FS, beta):
    """ attempts to analytically calculate Z from the sampling model (rho_#)
    checks for previous results data, reads it in, and then tosses the data if the hashes are not valid
    attempts to execute the analytical julia script, and if successful saves the output
    writes the combined old data and new data to the appropriate file
    """
    path_analytic = FS.path_analytic_rho
    path_src = FS.path_rho_model

    command = construct_command_dictionary()["analytical_sampling"]

    old_dict = {}

    if isfile(path_analytic):
        with open(path_analytic, 'r') as file:
            data = file.read()
            if len(data) > 1:
                old_dict = json.loads(data)

    validate_old_rho_data(old_dict, FS)
    command = command.format(F=path_src, T=beta)

    if compute(command, path_src, old_dict, beta):
        validate_old_model_data(old_dict, FS)
        with open(path_analytic, 'w') as file:
            json.dump(old_dict, file)

    return


def analytic_of_original_coupled_model(FS, beta):
    """ attempts to analytically calculate Z from the original model
    checks for previous results data, reads it in, and then tosses the data if the hashes are not valid
    attempts to execute the analytical julia script, and if successful saves the output
    writes the combined old data and new data to the appropriate file
    """
    path_analytic = FS.path_analytic_orig
    path_src = FS.path_orig_model

    command = construct_command_dictionary()["analytical_coupled"]

    old_dict = {}

    if isfile(path_analytic):
        with open(path_analytic, 'r') as file:
            data = file.read()
            if len(data) > 1:
                old_dict = json.loads(data)

    validate_old_model_data(old_dict, FS)
    command = command.format(F=path_src, T=beta)

    if compute(command, path_src, old_dict, beta):
        validate_old_model_data(old_dict, FS)
        with open(path_analytic, 'w') as file:
            json.dump(old_dict, file)

    return


def sos_of_coupled_model(FS, basis_size, beta):
    """ attempts to calculate Z using sos
    checks for previous results data, reads it in, and then tosses the data if the hashes are not valid
    attempts to execute the sos julia script, and if successful saves the output
    writes the combined old data and new data to the appropriate file
    """
    old_dict = {}

    path_sos = FS.template_sos_vib.format(B=basis_size)

    command = construct_command_dictionary()["sos"]

    if isfile(path_sos):
        with open(path_sos, 'r') as file:
            data = file.read()
            if len(data) > 1:
                old_dict = json.loads(data)

    validate_old_model_data(old_dict, FS)
    command = command.format(F=FS.path_vib_model, T=beta, B=basis_size)

    if compute(command, path_src, old_dict, beta):
        validate_old_model_data(old_dict, FS)
        with open(path_sos, 'w') as file:
            json.dump(old_dict, file)

    return


def trotter_of_coupled_model(FS, nbeads, basis_size, beta):
    """ attempts to calculate Z using sos , including the trotter error
    checks for previous results data, reads it in, and then tosses the data if the hashes are not valid
    attempts to execute the trotter julia script, and if successful saves the output
    writes the combined old data and new data to the appropriate file
    """
    old_dict = {}

    path_trotter = FS.template_trotter_vib.format(P=nbeads, B=basis_size)

    command = construct_command_dictionary()["trotter"]

    if isfile(path_trotter):
        with open(path_trotter, 'r') as file:
            data = file.read()
            if len(data) > 1:
                old_dict = json.loads(data)

    validate_old_model_data(old_dict, FS)
    command = command.format(F=FS.path_vib_model, T=beta, B=basis_size, P=nbeads)

    if compute(command, path_src, old_dict, beta):
        validate_old_model_data(old_dict, FS)
        with open(path_trotter, 'w') as file:
            json.dump(old_dict, file)

    return


def iterate_method(FS, n_iterations=50):
    """ this is just a wrapper for the iterate method at the moment
    it doesn't check for old data - it just regenerates the output every time
    """
    path_iterate = join(FS.path_vib_params, "iterative_model.json")
    print(path_iterate)

    command = construct_command_dictionary()["iterate"]
    command = command.format(F=FS.path_vib_model, P=path_iterate, N=n_iterations)

    print("command", command)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (out, error) = p.communicate()
    print(out.decode())
    print(error.decode())
    # we assume there were no issues!
    # TODO - this should be fixed after the iterative method has been finalized

    # we need to modify the json file to get it into a sampling system format
    vIO.remove_coupling_from_model(path_iterate, path_iterate)
    return


def prepare_julia():
    """preform any tasks necessary to setup the environment for executing julia code"""
    cmd = ['/usr/bin/modulecmd', 'python', 'load', 'julia/1.0.0']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, error = p.communicate()
    exec(out)
    return


def construct_command_dictionary():
    """x"""

    # old_path_toolkit = "/.julia/v0.6/VibronicToolkit/bin/"

    # ------------------------------------------------------------------------------------------
    """
    TODO - find a more elegant way to do this, could use the TOML package but it would be desirable not to have to keep adding additional packages
    TOML package - https://pypi.org/project/toml/
    this is a temporary workaround as the julia code is being modified for 1.0 and in development mode
    """
    path_julia_dev = os.path.expanduser("~/.julia/environments/v1.0/Manifest.toml")

    with open(path_julia_dev, 'r') as file:
        data = file.readlines()

    for idx, line in enumerate(data):
        if "[[VibronicToolkit]]" in line:
            path_toolkit = data[idx+2].split("=")[1].strip().replace('"', '')
            path_toolkit = os.path.normpath(path_toolkit)
            assert os.path.exists(path_toolkit)

    julia = "julia " + path_toolkit + "/bin/"
    # ------------------------------------------------------------------------------------------

    a_c = julia + "analytical_coupled.jl --conf {F:} --beta {T:}"
    a_s = julia + "analytical_sampling.jl --conf {F:} --beta {T:}"
    s = julia + "sos.jl --conf {F:} --beta {T:} --basis-size {B:}"
    t = julia + "trotter.jl --conf {F:} --beta {T:} --basis-size {B:} --num-links {P:}"
    x = julia + "sampling.jl --conf {F:} --beta {T:} --num-links {P:} --num-samples {X:}"
    i = julia + "iterate.jl --conf {F:} --path {P:} --num-iter {N:}"

    d = {"analytical_coupled": a_c,
         "analytical_sampling": a_s,
         "sos": s,
         "trotter": t,
         "sampling": x,
         "iterate": i,
         }
    return d


def main():
    """x"""

    assert len(sys.argv) in [4, 5], "wrong number of args"
    assert sys.argv[3].isnumeric() and int(sys.argv[3]) >= 0, "rho id is invalid"

    temperature = 300.0 if (len(sys.argv) == 4) else float(sys.argv[4])
    beta = constants.beta(temperature)

    prepare_julia()

    cmd_dict = construct_command_dictionary()

    path_root = sys.argv[1]
    id_data = int(sys.argv[2])
    id_rho = int(sys.argv[3])

    FS = fs.FileStructure(path_root, id_data, id_rho)
    FS.generate_model_hashes()

    # analytic_of_original_coupled_model(FS, beta)
    analytic_of_sampling_model(FS, beta)
    # sos_of_coupled_model(FS, beta)


if (__name__ == "__main__"):
    main()
