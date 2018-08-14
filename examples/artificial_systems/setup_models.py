# setup_models.py - creates all the directories and copies in the necessary files

# system imports
import shutil
import time
import sys
from os.path import join, abspath, dirname, isfile

# third party imports
import numpy as np

# local imports
import context
import systems
from pibronic.vibronic import vIO, VMK
import pibronic.data.file_structure as fs
sys.path.insert(1, abspath(__file__))  # TODO - fix this import in the future
from submit_jobs_to_server import iterative_method_wrapper


def generate_modified_energy(FS, path_rho_src):
    """ generate new E^{aa} values for the iterative model, as an attempt to solve the weight issue for uncoupled models """

    iterative_model = vIO.load_diagonal_model_from_JSON(path_rho_src)
    assert VMK.G2 not in iterative_model, "doesn't support quadratic terms at the moment"
    Ai, Ni = iterative_model[VMK.A], iterative_model[VMK.N]
    w = iterative_model[VMK.w]
    lin = iterative_model[VMK.G1]

    # generate the oscillator minimum's
    minimums = np.zeros((Ai, Ni))
    for a in range(Ai):
        minimums[a, :] = np.divide(-lin[:, a],  w[:])
        # print("m", minimums[a, :], sep='\n')
        # print("direct", -lin[:, a] / w[:], sep='\n')

    # print("the da's", minimums.shape, minimums, sep="\n")

    coupled_model = vIO.load_model_from_JSON(FS.path_vib_model)
    A, N = coupled_model[VMK.A], coupled_model[VMK.N]
    assert VMK.G2 not in coupled_model, "doesn't support quadratic terms at the moment"
    E = coupled_model[VMK.E]
    w = coupled_model[VMK.w]
    lin = coupled_model[VMK.G1]
    # quad = coupled_model[VMK.G2]

    new_energy_values = np.zeros(A)

    for a in range(A):
        q = minimums[a, :]  # we evaluate the R/q point

        # compute the harmonic oscillator contribution
        ho = np.zeros((A, A))
        np.fill_diagonal(ho, np.sum(w * pow(q, 2)))
        ho *= 0.5
        # print("the H.O.", ho.shape, ho, sep="\n")

        V = np.zeros((A, A))
        V[:] += E
        # print("V with energy", V, sep="\n")
        V[:] += ho
        # print("V with H.O.", V, sep="\n")

        for b1 in range(A):
            for b2 in range(A):
                V[b1, b2] += np.sum(lin[:, b1, b2] * q)
        # print("V with linear terms", V, sep="\n")

        evals = np.linalg.eigvalsh(V)
        lowest_eigvals = min(evals)
        # print(evals, lowest_eigvals)
        new_energy_values[a] = lowest_eigvals

    return new_energy_values


def create_sampling_distributions(FS, name):
    """ copies the original_coupled_model.json file into the /rho_#/sampling_model.json
    if use_original_model is True
    otherwise it simply uses the diagonal of coupled_model.json as the sampling model"""
    # vIO.create_harmonic_model(FS)

    # for the first sampling model we want to use the basic sampling model
    #  which is simply the diagonal of the Hamiltonian (not the original)
    id_rho = 0
    FS.change_rho(id_rho)
    vIO.create_basic_diagonal_model(FS)

    # for the second sampling model we want to use the original_coupled_model as our sampling model
    #  which is simply the diagonal of the Hamiltonian (not the original)
    id_rho = 1
    FS.change_rho(id_rho)
    source = FS.path_orig_model
    dest = FS.path_rho_model
    vIO.remove_coupling_from_model(source, dest)
    # I believe we need to use remove_coupling_from_model() to change the dimensions of
    # the orig_model, even though it is already diagonal, rho_model's have only
    # one surface dimension not 2

    iterate = False
    # for the third sampling model we want to use the iterative method
    if iterate:
        id_rho = 2
        FS.change_rho(id_rho)
        iterative_method_wrapper(id_data=FS.id_data)
        # copy to rho 2
        time.sleep(2)  # fixes a race condition one time, which I encountered one time
        path_iterate = join(FS.path_vib_params, "iterative_model.json")
        path_rho_2_model = shutil.copy(path_iterate, FS.path_rho_model)

        id_rho = 3
        FS.change_rho(id_rho)
        E = generate_modified_energy(FS, path_rho_2_model)
        # copy to rho 3
        path_iterate = join(FS.path_vib_params, "iterative_model.json")
        shutil.copy(path_iterate, FS.path_rho_model)
        # modify model
        model = vIO.load_diagonal_model_from_JSON(FS.path_rho_model)
        model[VMK.E] = E
        vIO.save_diagonal_model_to_JSON(FS.path_rho_model, model)

    # TODO - write function for vIO that just converts the array's between sizes
    # and throws an error if the input is not diagonal
    return


def unitary_transformation(FS, force_new=False):
    """ backup the old coupled_model.json
    and create a new one through a unitary transformation
    """
    if not force_new and isfile(FS.path_ortho_mat):
        # if we already have a orthonormal matrix then use that one
        vIO.create_fake_coupled_model(FS, transformation_matrix="re-use")
    else:
        # otherwise create a new one
        param = systems.get_tuning_parameter(FS.id_data)
        vIO.create_fake_coupled_model(FS, tuning_parameter=param)
    return


def parse_input_mctdh_files_into_directories(FS, system_name):
    """ fill the directories with the appropriate data so we can run simulations
    starts from MCTDH *.op files and creates the relevant *.json files
    the MCTDH files are also easier to read when the parameter sizes are larger"""

    # where the MCTDH input files for each model are stored
    dir_src = join(abspath(dirname(__file__)), "input_mctdh")
    path_src = join(dir_src, f"model_{system_name:s}.op")

    path_dst = shutil.copy(path_src, FS.path_es)

    # create the coupled_model.json file from the system_[1-6].op file
    vIO.create_coupling_from_op_file(FS, path_dst)
    return


def copy_input_json_files_into_directories(FS, system_name):
    """ fill the directories with the appropriate data so we can run simulations
    starts from the preconstructed *.json file"""

    # where the .json input files for each model are stored
    dir_src = join(abspath(dirname(__file__)), "input_json")
    path_src = join(dir_src, f"model_{system_name:s}.json")

    # copy the .json file into the appropriate directory
    shutil.copy(path_src, FS.path_vib_model)
    return


def prepare_model(system_name, id_data=0, root=None):
    """ wrapper function """
    systems.assert_id_data_is_valid(id_data)

    if root is None:
        root = context.choose_root_folder()

    # instantiate the FileStructure object which creates the directories
    FS = fs.FileStructure(root, id_data, 0)

    # either way will work
    if True:
        copy_input_json_files_into_directories(FS, system_name)
    else:
        parse_input_mctdh_files_into_directories(FS, system_name)

    unitary_transformation(FS)

    # create the simple rho's
    # which are the diagonal of the transformed Hamiltonian (id_rho = 0)
    # and the original_coupled_model (id_rho = 1)
    # and the iterative method (id_rho = 2)
    # and the weight corrected iterative method (id_rho = 3)
    create_sampling_distributions(FS, system_name)

    return


def automate_prepare_model():
    """ convenience/wrapper function
    loops over all the models we want to prepare
    """
    for name in systems.name_lst:
        for id_data in systems.id_dict[name]:
            prepare_model(name, id_data=id_data)
    return


if (__name__ == "__main__"):
    automate_prepare_model()
