#!python3

# system imports
import subprocess
import json
import sys
import os

# third party imports

# local imports
from pibronic.data import file_structure as fs
from pibronic import constants
from pibronic.log_conf import log


def compute(command, old_dict):
    """x"""
    log.flow("Computing")
    log.flow(command)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (out, error) = p.communicate()
    # out = ""
    print(out.decode())
    print(error.decode())
    rawOutput = ''.join(list(out.decode()))
    lineList = [string for string in rawOutput.split("\n") if string is not '']
    output_dict = dict(line.split(":") for line in lineList)

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

    new_dict = {str(temperature): temp_dict}

    old_dict.update(new_dict)
    return


def validate_old_data(old_dict, FS):
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


def analytic_of_sampling_model(FS, command, beta):
    """x"""
    path_analytic = FS.path_analytic_rho
    path_original = FS.path_rho_model

    old_dict = {}

    if os.path.isfile(path_analytic):
        with open(path_analytic, 'r') as file:
            data = file.read()
            if len(data) > 1:
                old_dict = json.loads(data)

    validate_old_data(old_dict, FS)
    command = command.format(F=path_original, T=beta)
    compute(command, old_dict)

    with open(path_analytic, 'w') as file:
        json.dump(old_dict, file)
    return


def analytic_of_original_coupled_model(FS, command, beta):
    """x"""
    path_analytic = FS.path_vib_params + "original_analytic_results.txt"
    path_original = FS.path_vib_params + "original_coupled_model.json"

    old_dict = {}

    if os.path.isfile(path_analytic):
        with open(path_analytic, 'r') as file:
            data = file.read()
            if len(data) > 1:
                old_dict = json.loads(data)

    validate_old_data(old_dict, FS)
    command = command.format(F=path_original, T=beta)
    compute(command, old_dict)

    with open(path_analytic, 'w') as file:
        json.dump(old_dict, file)

    return


def sos_of_coupled_model(FS, command, beta):
    """x"""
    old_dict = {}
    basis_size = 80

    path_sos = FS.template_sos_vib.format(B=basis_size)

    # is this a mistake? why check for vib then open rho ??
    if os.path.isfile(path_sos):
        with open(path_sos, 'r') as file:
            data = file.read()
            if len(data) > 1:
                old_dict = json.loads(data)

    command = command.format(F=FS.path_vib_model, T=beta, B=basis_size)
    compute(command, old_dict)

    validate_old_data(old_dict, FS)
    with open(path_sos, 'w') as file:
        json.dump(old_dict, file)

    return


def prepare_julia():
    """preform any tasks necessary to setup the environment for executing julia code"""
    cmd = ['/usr/bin/modulecmd', 'python', 'load', 'julia/0.6.3']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, error = p.communicate()
    exec(out)
    return


def construct_command_dictionary():
    """x"""
    julia = "julia ~/.julia/v0.6/VibronicToolkit-Integrated/bin/"
    a_c = julia + "analytical_coupled.jl --conf {F:} --beta {T:}"
    a_s = julia + "analytical_sampling.jl --conf {F:} --beta {T:}"
    s = julia + "sos.jl --conf {F:} --beta {T:} --basis-size {B:}"
    t = julia + "trotter.jl --conf {F:} --beta {T:} --basis-size {B:} --num-links {P:}"
    x = julia + "sampling.jl --conf {F:} --beta {T:} --num-links {P:} --num-samples {X:}"
    return {"analytical_coupled": a_c, "analytical_sampling": a_s, "sos": s, "trotter": t, "sampling": x}


if (__name__ == "__main__"):
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

    # analytic_of_original_coupled_model(FS, cmd_dict["analytical_coupled"], beta)
    analytic_of_sampling_model(FS, cmd_dict["analytical_sampling"], beta)
    # sos_of_coupled_model(FS, cmd_dict["sos"], beta)
