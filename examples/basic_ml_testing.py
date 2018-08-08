# minimal.py

# system imports
import os
from os.path import join
import shutil

# third party imports
import numpy as np
from numpy import float64 as F64

# local imports
from context import pibronic
import pibronic.data.file_structure as fs
import pibronic.data.file_name as fn
import pibronic.data.vibronic_model_io as vIO
from pibronic.pimc.minimal import BoxData, BoxResult, block_compute_gR_from_raw_samples, block_compute_rhoR_from_input_samples


def generate_basic_data_set(FS, P=12, T=300.00, J=0):
    """ This will generate two files:
    - P#_T#_J#_training_data_input.npz
    which holds the following key, value pairs
            number_of_samples: int
            input_r_values: np.array of size XNP, of data type np.float64

    - P#_T#_J#_training_data_g_output.npz
    which holds the following key, value pairs
            number_of_samples: int
            g: the output of the function g on the corresponding R values in the previous file

    The goal of the ML algorithm is to generate a
        root/data_set_#/rho_#/parameters/sampling_model.json
    file which matches the training_data_output as close as possible given the training_data_input
    """
    np.random.seed()
    # np.random.seed(429312)

    # load the relevant data
    data = BoxData.from_FileStructure(FS)

    data.samples = int(1e2)
    data.beads = P
    data.temperature = T
    data.block_size = int(1e2)
    data.blocks = data.samples // data.block_size

    # setup empty tensors, models, and constants
    data.preprocess()

    # store results here
    result = BoxResult(data=data)
    result.id_job = J
    result.path_root = FS.path_vib_results
    block_compute_gR_from_raw_samples(data, result)
    return data, result


def generate_rho_output_values(FS, data, result, id_rho=None):
    """
    Will generate the corresponding rho(R) values and save them to
    P#_T#_J#_training_data_rho_output.npz

    if provided with a id_rho that is different from the current one stored in date it will
    attempt to generate rho_output for that rho
    """

    if type(id_rho) is int and not id_rho == data.id_rho:
        # We need to compute on another rho
        data.id_rho = id_rho
        FS = fs.FileStructure(root, data)
        assert os.path.isfile(FS.path_rho_model), f"You need to create a sampling_model.json file at {FS.path_rho_model:}"
        data.path_rho_model = FS.path_rho_model
        data.initialize_models()

    # store the R values here
    input_R_values = np.empty(data.size['XNP'], dtype=F64)

    # read in the R values
    path = join(FS.path_vib_results, fn.training_data_input())
    path = path.format(P=data.beads, T=data.temperature, J=result.id_job)
    with np.load(path) as input_dict:
        assert input_dict["number_of_samples"] == input_R_values.shape[0], "wrong number of samples, i.e. X"
        input_R_values = input_dict["input_R_values"]

    result.path_root = FS.path_rho_results
    block_compute_rhoR_from_input_samples(data, result, input_R_values)

    # now save the rho(R) values
    path = join(FS.path_rho_results, fn.training_data_rho_output())
    path = path.format(P=data.beads, T=data.temperature, J=result.id_job)

    np.savez(path,
             number_of_samples=result.samples,
             rho=result.scaled_rho,
             )
    return


if (__name__ == "__main__"):
    # creates the output files in the /Pibronic/examples/ folder, which you can change if you want
    root = os.path.abspath(os.path.dirname(__file__))

    # create the FileStructure object which will create the directories need to organize the files
    FS = fs.FileStructure(root, 0, 0)

    # copy the provided fake_vibron file into the electronic structure directory
    new_path = shutil.copy(join(root, "fake_vibron.op"), FS.path_es)

    # create the coupled_model.json file from the fake_vibron.op file
    # this is not really necessary given that this is a test case
    # but we want to be able to replace the fake_vibron.op data with real data eventually
    vIO.create_coupling_from_op_file(FS, new_path)
    vIO.create_harmonic_model(FS)
    vIO.create_basic_diagonal_model(FS)

    # you can change the bead number to a smaller value if thats helpful, although it has to be at least 3, don't use a bigger value atm, that will just generate excess data
    P = 12
    T = 300.0  # leave the temperature at 300 for now
    # if you want to generate more data then you can use additional job numbers
    list_id_job = range(0, 1)

    for J in list_id_job:
        data, result = generate_basic_data_set(FS, P=P, T=T, J=J)
        generate_rho_output_values(FS, data, result)

    # ------------------------------------------------------------------------------------------
    # just a little example code

    # for a given rho number, such as 0
    id_rho = 0
    # we create a FileStructure obj
    FS = fs.FileStructure(root, 0, id_rho)
    # and we can open the files for a given job number, such as 0
    id_job = 0

    # this is how to open the training_data_input file
    path = join(FS.path_vib_results, fn.training_data_input())
    path = path.format(P=P, T=T, J=J)
    file_dict = np.load(path)
    print(f"The contents of {path:}")
    for k, v in file_dict.items():
        print(k, type(v), v.shape)
    # this is how to open the training_data_g_output file
    path = join(FS.path_vib_results, fn.training_data_g_output())
    path = path.format(P=P, T=T, J=J)
    print(f"\nThe contents of {path:}")
    file_dict = np.load(path)
    for k, v in file_dict.items():
        print(k, type(v), v.shape)
    # this is how to open the training_data_rho_output file
    path = join(FS.path_rho_results, fn.training_data_rho_output())
    path = path.format(P=P, T=T, J=J)
    print(f"\nThe contents of {path:}")
    file_dict = np.load(path)
    for k, v in file_dict.items():
        print(k, type(v), v.shape)

    # ------------------------------------------------------------------------------------------
