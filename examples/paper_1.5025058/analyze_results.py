# analyze_results.py - creates all the directories and copies in the necessary files

# system imports
import socket
import shutil
import glob
import os
from os.path import join

# third party imports

# local imports
import context
from context import pibronic
from pibronic.vibronic import vIO
from pibronic.stats import stats
import pibronic.data.file_structure as fs

# list of the names for each of the four system's
from context import system_names

# lists of the coupled model id's for each system
# in a dictionary, with the system's name as the key
from context import data_dict


def get_FS(root, id_data, id_rho):
    system_name = context.get_system_name(id_data)
    assert id_data in data_dict[system_name], f"invalid id_data ({id_data:d})"

    if root is None:
        root = context.choose_root_folder()

    # instantiate the FileStructure object which creates the directories
    return fs.FileStructure(root, id_data, id_rho)


def simple_wrapper(FS):
    # TODO - sort out which ones I want to use
    stats.statistical_analysis_of_pimc(FS.path_root, FS.id_data, FS.id_rho, method="basic")
    # stats.statistical_analysis_of_pimc(FS.path_root, FS.id_data, FS.id_rho, method="alpha")
    stats.jackknife_analysis_of_pimc(FS.path_root, FS.id_data, FS.id_rho, method="basic")
    # stats.jackknife_analysis_of_pimc(FS.path_root, FS.id_data, FS.id_rho, method="alpha")
    return


def automate_wrapper(name):
    dir_src = join(os.path.abspath(os.path.dirname(__file__)), "alternate_rhos")
    template = join(dir_src, "{:s}_D{:d}_R{:d}.json")

    for id_data in data_dict[name]:
        for id_rho in range(0, 10):
            FS = get_FS(None, id_data, id_rho)
            # TODO - this is a perfect example where FS should have a method which returns a bool
            # which tells if all the directories or necessary - WITHOUT CREATING THE DIRECTORIES
            if os.path.isfile(FS.path_rho_model):
                simple_wrapper(FS)
    return


if (__name__ == "__main__"):
    # automate_wrapper("superimposed")
    # automate_wrapper("displaced")
    # automate_wrapper("elevated")
    automate_wrapper("jahnteller")