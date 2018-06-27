# setup_models.py - creates all the directories and copies in the necessary files

# system imports
import socket
import shutil
import glob
import os
from os.path import join

# third party imports

# local imports
import context
from pibronic.vibronic import vIO
import pibronic.data.file_structure as fs

# list of the names for each of the four system's
from context import system_names

# lists of the coupled model id's for each system
# in a dictionary, with the system's name as the key
from context import data_dict


def parse_input_mctdh_files_into_directories(FS, system_name):
    """ fill the directories with the appropriate data so we can run simulations
    starts from MCTDH *.op files and creates the relevant *.json files """

    # where the MCTDH input files for each model are stored
    dir_src = join(os.path.abspath(os.path.dirname(__file__)), "input_mctdh")

    path_src = join(dir_src, f"{system_name:s}_{FS.id_data%10:d}.op")
    path_dst = shutil.copy(path_src, FS.path_es)

    # create the coupled_model.json file from the system_[1-6].op file
    # this is not necessary since we know what the *.json files will look like,
    # but simulating the full procedure is fine
    vIO.create_coupling_from_op_file(FS, path_dst)
    vIO.create_harmonic_model(FS)
    # this basic sampling model is simply the diagonal of the hamiltonian
    vIO.create_basic_sampling_model(FS)
    return


def copy_input_json_files_into_directories(FS, system_name):
    """ fill the directories with the appropriate data so we can run simulations
    starts from the preconstructed *.json file"""

    # where the .json input files for each model are stored
    dir_src = join(os.path.abspath(os.path.dirname(__file__)), "input_json")

    path_src = join(dir_src, f"{system_name:s}_{FS.id_data%10:d}.json")

    # copy the .json file into the appropriate directory
    shutil.copy(path_src, FS.path_vib_model)
    # this basic sampling model is simply the diagonal of the hamiltonian
    print(FS.path_rho_params)
    vIO.create_basic_sampling_model(FS)
    return


def copy_alternate_rhos_into_directories(FS, system_name):
    """ scan files in root/alternate_rhos/ directory
    make directories for and copy in all relevant files """

    # where the .json input files for each sampling distribution are stored
    dir_src = join(os.path.abspath(os.path.dirname(__file__)), "alternate_rhos")

    # collect all possible alternative rho's
    globPath = join(dir_src, f"{system_name:s}_D{FS.id_data%10:d}_R*.json")
    lst_files = [file for file in glob.glob(globPath)]

    for file in lst_files:
        # extract the id
        id_rho = int(file.split("_R")[1].split(".json")[0])
        # create the directories
        FS.change_rho(id_rho)
        # copy the file over
        shutil.copy(file, FS.path_rho_model)

    return


def prepare_model(system_name, id_data=0, root=None):
    """ wrapper function """

    assert id_data in data_dict[system_name], f"invalid id_data ({id_data:d})"

    if root is None:
        root = context.choose_root_folder()

    # instantiate the FileStructure object which creates the directories
    FS = fs.FileStructure(root, id_data, 0)

    # either way will work
    if True:
        copy_input_json_files_into_directories(FS, system_name)
    else:
        parse_input_mctdh_files_into_directories(FS, system_name)

    # create all possible rho's
    copy_alternate_rhos_into_directories(FS, system_name)

    return


def automate_prepare_model():
    """ convience/wrapper function
    loops over all the models we want to prepare
    """

    for system in system_names:
        for id_data in data_dict[system]:
            prepare_model(system, id_data=id_data)

    return


if (__name__ == "__main__"):

    # prepare_model("superimposed", id_data=11)
    automate_prepare_model()
