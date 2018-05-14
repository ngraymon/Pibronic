"""A first test of the full server stack
can't be run by pytest or travis becuase it requires SLURM """

# system imports
import os
from os.path import join
import shutil

# local imports
from context import pibronic
import pibronic.data.vibronic_model_io as vIO
import pibronic.data.file_structure as fs
from pibronic.log_conf import log
from pibronic.server import job_boss

# third party imports
# import pytest

test_path = "/work/ngraymon/pimc/testing/"


def setup_data_set_for_execution(id_data=0, id_rho=0):
    """x"""
    FS = fs.FileStructure(test_path, id_data, id_rho)

    # simple check
    assert os.path.exists(join(test_path, f"data_set_{id_data:}/rho_{id_rho:}/"))

    path_src = os.path.abspath(os.path.dirname(__file__))
    # copy the provided fake_vibron file into the electronic structure directory
    new_path = shutil.copyfile(join(path_src, f"fake_vibron_{id_data:d}.op"),
                               join(FS.path_es, "fake_vibron.op")
                               )

    # create the coupled_model.json file from the fake_vibron.op file
    # this is not really necessary given that this is a test case
    # but we want to be able to replace the fake_vibron.op data with real data eventually
    vIO.create_coupling_from_op_file(FS, new_path)
    vIO.create_harmonic_model(FS)
    vIO.create_basic_sampling_model(FS)

    return FS


if (__name__ == "__main__"):
    """x"""

    # select execution parameters
    id_data = 0
    id_rho = 0

    # FS = setup_data_set_for_execution(id_data=id_data, id_rho=id_rho)
    FS = fs.FileStructure(test_path, id_data, id_rho)
    log.flow("Finished setting up execution directory")
    # ---------------------------------------------------------------------------------------------
    # SUBMIT PIMC JOBS

    temperature_list = [250., 275., 300., 325., 350., ]
    bead_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, ]

    N, A = vIO.get_nmode_nsurf_from_coupled_model(FS)

    # this is the minimum amount of data needed to run an execution
    parameter_dictionary = {
        "number_of_samples": int(1e4),
        "number_of_states": A,
        "number_of_modes": N,
        "bead_list": bead_list,
        "temperature_list": temperature_list,
        "delta_beta": None,  # we shouldn't need this for a non PM calculation
        "id_data": id_data,
        "id_rho": id_rho,
    }

    # create an execution object
    engine = job_boss.PimcSubmissionClass(FS, parameter_dictionary)

    log.flow("Submitting jobs to server")
    engine.submit_jobs()

    log.flow("Finished PIMC calculation")
    # ------------------------------------------------------------------------------------------
    # just a little example code
