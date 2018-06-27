#! /usr/bin/env python3
"""
Usuage: {scriptName \#}.format(scriptName = sys.argv[0])

"""

# system imports
from pathlib import Path
import subprocess
import shutil
import sys
import os
from os.path import join

# third party imports

# local imports
# sys.path.append('/home/ngraymon/pibronic/')
from . import constants
from .log_conf import log
from .vibronic import vIO
from .vibronic import electronic_structure as ES
from .data import file_structure
from . import pimc
from .server import job_boss

# source for generating vibronic models
sys.path.append("..")
from examples import input_files

# -------------------------------------------------------------------------------------------------
# REFERENCE DATA


model_dict = {
    0:    ("acetonitrile", 200),
    1:    ("ammonia", 201),
    2:    ("boron_trifluoride", 202),
    3:    ("formaldehyde", 203),
    4:    ("methane", 204),
    5:    ("formamide", 205),
    6:    ("formic_acid", 206),
    7:    ("hydrogen_peroxide", 207),
    8:    ("water", 208),
    9:    ("pyridine", 209),
    10:   ("furan", 210),
    11:   ("trichloroethylene", 211),
    12:   ("chloroethylene", 212),
    13:   ("acrolein", 213),
    14:   ("acrylonitrile", 214),
    15:   ("cis_12_dichloroethylene", 215),
    16:   ("trans_12_dichloroethylene", 216),
    17:   ("11_dichloroethylene", 217),
    # 18:   ("", 218),
    # 19:   ("", 219),
    # 20:   ("", 220),
    # 21:   ("", 221),
    }


def copyInput(FS, file_name):
    """x"""
    path_root = os.path.abspath(input_files.__file__)

    src_params = join(path_root, file_name, "_params.txt")
    src_zmat = join(path_root, file_name, "_zmat.txt")

    dst_params = join(FS.path_es, file_name, "_params.txt")
    dst_zmat = join(FS.path_es, file_name, "_zmat.txt")

    shutil.copyfile(src_params, dst_params)
    shutil.copyfile(src_zmat, dst_zmat)

    s = "Successfully created input files {:s} {:s}"
    log.flow(s.format(dst_params, dst_zmat))


def generate_analytical_results(root, id_data, id_rho, temperature_list):
    """generate analytical results using julia"""

    # need a check to see if analytical results exist
    return  # hacks for now

    for T in temperature_list:
        # this should also be generalized
        command = "python3 ~/pibronic/pibronic/julia_wrapper.py {:s} {:d} {:d} {:.2f}\n"
        # should replace this with a function call that isn't dependent on our server
        # sshProcess = subprocess.Popen(["ssh", "-t", "dev002"],
        sshProcess = subprocess.Popen(["srun", "--pty"],  # dev002 is out of commission
                                      universal_newlines=True,
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      )

        sshProcess.stdin.write(command.format(root, id_data, id_rho, T))
        sshProcess.stdin.close()
        log.flow("Currently generating analytical parameters at Temperature {:.2f}".format(T))
        sshProcess.wait(timeout=120)
    return


def execute(file_path, id_model):
    """x"""
    if id_model not in model_dict.keys():
        raise Exception("Argument must be one of " + str(model_dict.keys()))

    molecule_tuple = model_dict[id_model]
    molecule_name = molecule_tuple[0]
    id_data = molecule_tuple[1]
    id_rho = 0  # for now we just do the simple option

    # ---------------------------------------------------------------------------------------------
    # CREATE THE DIRECTORIES

    files = file_structure.FileStructure('/work/ngraymon/pimc/', id_data)
    if not files.directories_exist():
        files.make_directories()
    else:
        log.info("directories already exist")

    # ---------------------------------------------------------------------------------------------
    # COPY THE INPUT FILES
    copyInput(files, molecule_name)

    # ---------------------------------------------------------------------------------------------
    # CREATE THE VIBRONIC MODEL

    path_result = files.path_es + molecule_name + "_vibron.h"
    if (not os.path.isfile(path_result)):
        log.flow("{:s} file not found, attempting to calculate vibronic model".format(path_result))
        job = ES.VibronExecutionClass(molecule_name, files.path_es)
        job.calculate_vibronic_model()
        log.flow("Finished calculating vibronic model")
    else:
        log.flow("Vibronic model already calculated at: {:s}".format(path_result))

    # ---------------------------------------------------------------------------------------------
    # PREPARE INPUT FOR PIMC CALCULATION

    # extract the model from *_vibron.h and store in json format
    path_model_coupled = vIO.create_coupling_from_h_file(path_result)

    # Create the harmonic version
    path_model_harmonic = vIO.create_harmonic_model(id_data)

    # Create the sampling model
    path_model_sampling = vIO.create_basic_sampling_model(id_data, id_rho)

    # Copy to rho_0 to represent the simple sampling model
    # path_model_sampling = files.path_data + "rho_0/parameters/sampling_model.json"
    # shutil.copyfile(path_model_harmonic, path_model_sampling)

    temperature_list = [250., 275., 300., 325., 350.]
    bead_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, ]

    generate_analytical_results(files.path_root, id_data, id_rho, temperature_list)
    log.flow("Finished preparing input for PIMC calculation")
    # ---------------------------------------------------------------------------------------------
    # SUBMIT PIMC JOBS

    A, N = vIO.extract_dimensions_of_coupled_model(FS=files)

    # this is the minimum amount of data needed to run an execution
    parameter_dictionary = {
        "number_of_samples": int(1e6),
        "number_of_states": A,
        "number_of_modes": N,
        "bead_list": bead_list,
        "temperature_list": temperature_list,
        "delta_beta": constants.delta_beta,
        "id_data": id_data,
        "id_rho": id_rho,
    }

    # create an execution object
    engine = job_boss.PimcSubmissionClass(files, parameter_dictionary)

    engine.submit_jobs()

    log.flow("Finished PIMC calculation")
    # ---------------------------------------------------------------------------------------------
    # RUN JACKKNIFE ON RESULTS TO GENERATE THERMO FILES

    # not fully integrated into tool chain yet

    log.flow("Finished running Jackknife calculations")
    # ---------------------------------------------------------------------------------------------
    # Plot results!

    # not fully integrated into tool chain yet

    log.flow("Finished plotting Z")
    # ---------------------------------------------------------------------------------------------
    # All done!

    return


# shows docstring if --help is the second argument
if (__name__ == '__main__'):
    assert(len(sys.argv) == 2)  # make sure we provide an arg

    if sys.argv[1] == '--help':
        print(__doc__)

    execute(int(sys.argv[1]))
