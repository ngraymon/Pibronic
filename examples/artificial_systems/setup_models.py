# setup_models.py - creates all the directories and copies in the necessary files

# system imports
import shutil
import glob
import os
from os.path import join

# third party imports

# local imports
import context
import systems
from pibronic.vibronic import vIO
import pibronic.data.file_structure as fs


def create_sampling_distribution(FS, use_original_model=False):
    """ copys the original_coupled_model.json file into the /rho_#/sampling_model.json
    if use_original_model is True
    otherwise it simply uses the diagonal of coupled_model.json as the sampling model"""
    vIO.create_harmonic_model(FS)

    if use_original_model:
        # use the original_coupled_model as our ampling model
        source = FS.path_orig_model
        dest = FS.path_rho_model
        vIO.remove_coupling_from_model(source, dest)
        # I believe we need to use remove_coupling_from_model() to change the dimensions of
        # the orig_model, even though it is already diagonal, rho_model's have only
        # one surface dimension not 2

        # TODO - write function for vIO that just converts the array's between sizes
        # and throws an error if the input is not diagonal
    else:
        # this basic sampling model is simply the diagonal of the hamiltonian (not the original)
        vIO.create_basic_sampling_model(FS)

    return


def unitary_transformation(FS):
    """ backup the old coupled_model.json
    and create a new one through a unitary transformation
    """
    if os.path.isfile(FS.path_ortho_mat):
        # if we already have a orthonormal matrix then use that one
        vIO.create_fake_coupled_model(FS, transformation_matrix="re-use")
    else:
        # otherwise create a new one
        vIO.create_fake_coupled_model(FS, tuning_parameter=0.1)
    return


def parse_input_mctdh_files_into_directories(FS, system_name):
    """ fill the directories with the appropriate data so we can run simulations
    starts from MCTDH *.op files and creates the relevant *.json files
    the MCTDH files are also easier to read when the parameter sizes are larger"""

    # where the MCTDH input files for each model are stored
    dir_src = join(os.path.abspath(os.path.dirname(__file__)), "input_mctdh")

    path_src = join(dir_src, f"model_D{FS.id_data:d}.op")
    path_dst = shutil.copy(path_src, FS.path_es)

    # create the coupled_model.json file from the system_[1-6].op file
    vIO.create_coupling_from_op_file(FS, path_dst)
    return


def copy_input_json_files_into_directories(FS, system_name):
    """ fill the directories with the appropriate data so we can run simulations
    starts from the preconstructed *.json file"""

    # where the .json input files for each model are stored
    dir_src = join(os.path.abspath(os.path.dirname(__file__)), "input_json")
    path_src = join(dir_src, f"model_D{FS.id_data:d}.json")

    # copy the .json file into the appropriate directory
    shutil.copy(path_src, FS.path_vib_model)
    return


def copy_alternate_rhos_into_directories(FS, system_name):
    """ scan files in root/alternate_rhos/ directory
    make directories for and copy in all relevant files """

    # where the .json input files for each sampling distribution are stored
    dir_src = join(os.path.abspath(os.path.dirname(__file__)), "alternate_rhos")

    # collect all possible alternative rho's
    globPath = join(dir_src, f"model_D{FS.id_data:d}_R*.json")
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
    systems.assert_id_data_is_valid(id_data)

    if root is None:
        root = context.choose_root_folder()

    # instantiate the FileStructure object which creates the directories
    FS = fs.FileStructure(root, id_data, 0)

    # either way will work
    if False:
        copy_input_json_files_into_directories(FS, system_name)
    else:
        parse_input_mctdh_files_into_directories(FS, system_name)

    unitary_transformation(FS)

    create_sampling_distribution(FS, use_original_model=True)

    # create all possible rho's
    copy_alternate_rhos_into_directories(FS, system_name)

    return


def automate_prepare_model():
    """ convience/wrapper function
    loops over all the models we want to prepare
    """

    for name in systems.name_lst:
        for id_data in systems.id_dict[name]:
            prepare_model(name, id_data=id_data)

    return


if (__name__ == "__main__"):
    automate_prepare_model()
