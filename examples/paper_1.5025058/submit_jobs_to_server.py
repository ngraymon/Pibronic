# setup_models.py - creates all the directories and copies in the necessary files

# system imports
import subprocess
import socket
import shutil
import glob
import os
from os.path import join

# third party imports

# local imports
import context
from pibronic import julia_wrapper
from pibronic import constants
from pibronic.log_conf import log
from pibronic.vibronic import vIO
from pibronic.server import job_boss
import pibronic.data.file_structure as fs

# list of the names for each of the four system's
from context import system_names

# lists of the coupled model id's for each system
# in a dictionary, with the system's name as the key
from context import data_dict

# list of hostnames for which we will store results in the /work/* directories
from context import list_of_server_hostnames


def get_FS(root, id_data, id_rho):
    system_name = context.get_system_name(id_data)
    assert id_data in data_dict[system_name], f"invalid id_data ({id_data:d})"

    if root is None:
        root = context.choose_root_folder()

    # instantiate the FileStructure object which creates the directories
    return fs.FileStructure(root, id_data, id_rho)


def generate_analytical_results(root, id_data, id_rho, temperature_list):
    """generate analytical results using julia"""

    # TODO - should have a check to see if the analytical results already exists
    # and also should check that the hash values match

    path_script = os.path.abspath(julia_wrapper.__file__)

    for T in temperature_list:
        # this should also be generalized
        log.flow("About to generate analytical parameters at Temperature {:.2f}".format(T))

        cmd = ("srun"
               f" --job-name=analytc_D{id_data:d}_R{id_rho:d}_T{T:.2f}"
               " python3"
               f" '{path_script:s}'"
               f" {root:s}"
               f" {id_data:d}"
               f" {id_rho:d}"
               f" {T:.2f}"
               )

        p = subprocess.Popen(cmd,
                             universal_newlines=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=True,
                             )
        # (out, error) = p.communicate()
        # print(out)
        # print(error)
    return


def simple_pimc_wrapper(root=None, id_data=11, id_rho=0):
    """ submits PIMC job to the server """
    FS = get_FS(root, id_data, id_rho)

    # lst_P = [12, ]
    lst_P = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, ]
    lst_T = [300.00, ]
    # lst_T = [250., 275., 300., 325., 350., ]

    generate_analytical_results(FS.path_root, id_data, id_rho, lst_T)
    A, N = vIO.extract_dimensions_of_coupled_model(FS=FS)

    # this is the minimum amount of data needed to run an execution
    parameter_dictionary = {
        "number_of_samples": int(1e6),
        "number_of_states": A,
        "number_of_modes": N,
        "bead_list": lst_P,
        "temperature_list": lst_T,
        "delta_beta": constants.delta_beta,
        "id_data": id_data,
        "id_rho": id_rho,
    }

    # create an execution object
    engine = job_boss.PimcSubmissionClass(FS, parameter_dictionary)

    engine.submit_jobs()

    return


def automate_wrapper(name):
    """ loops over the data sets and different rhos submiting PIMC jobs for each one  """
    dir_src = join(os.path.abspath(os.path.dirname(__file__)), "alternate_rhos")
    template = join(dir_src, "{:s}_D{:d}_R{:d}.json")

    for id_data in data_dict[name]:
        for id_rho in range(0, 10):
            # make sure that rho is a number corresponding to an already chosen sampling model
            if os.path.isfile(template.format(name, id_data%10, id_rho)):
                simple_pimc_wrapper(id_data=id_data, id_rho=id_rho)
    return


if (__name__ == "__main__"):
    # simple_wrapper()
    automate_wrapper("superimposed")
    automate_wrapper("displaced")
    automate_wrapper("elevated")
    automate_wrapper("jahnteller")
