"""A basic test of the pimc module
"""

# system imports
import os
import inspect


# local imports
from ..context import pibronic
from pibronic import pimc
import pibronic.data.file_structure as fs

# third party imports
import pytest
import numpy as np
from numpy import float64 as F64


@pytest.fixture()
def path():
    # does this work when deployed?
    return os.path.join(os.path.dirname(os.path.dirname(inspect.getfile(pibronic))), "tests/test_models/")


@pytest.fixture()
def FS(path):
    return fs.FileStructure(path, 0, id_rho=0)


@pytest.fixture()
def random_model():
    A = 3
    N = 4
    B = 10
    P = 15
    model = pimc.ModelClass(states=A, modes=N)
    model.energy = np.random.rand(A)
    model.omega = np.random.rand(N)
    model.linear = np.random.rand(N, A, A)
    model.quadratic = np.random.rand(N, N, A, A)
    model.size = {'A': (A),
                  'AN': (A, N),
                  'BPA': (B, P, A),
                  'BPAN': (B, P, A, N),
                  'BPAA': (B, P, A, A),
                  'BP': (B, P),
                  }
    model.delta_weight = np.zeros(model.size['A'], dtype=F64)
    return model


def test_TemperatureDependentClass(FS, random_model):
    """very basic test just to ensure minimal requirements are met, doesn't check math output"""
    tau = pibronic.constants.beta(300.00)
    pibronic.pimc.pimc.TemperatureDependentClass(random_model, tau)
    return


@pytest.fixture(params=[(0, 0), (0, 1), (1, 0), (1, 1)])
def data(request):
    data = pimc.BoxData()
    data.id_data = request.param[0]
    data.id_rho = request.param[1]
    return data


@pytest.fixture()
def FS(path, data):
    return fs.FileStructure.from_boxdata(path, data)


def test_simple_block_compute(FS, data):
    samples = int(1e1)
    block_size = int(1e1)
    np.random.seed(242351)  # pick our seed

    data.path_vib_model = FS.path_vib_model
    data.path_rho_model = FS.path_rho_model

    FS.generate_model_hashes()
    data.hash_vib = FS.hash_vib
    data.hash_rho = FS.hash_rho

    data.states = 2
    data.modes = 2

    data.samples = samples
    data.beads = 12
    data.temperature = 300.0
    data.blocks = samples // block_size
    data.block_size = block_size

    # setup empty tensors, models, and constants
    data.preprocess()

    # store results here
    results = pimc.BoxResult(data=data)
    results.path_root = FS.path_rho_results

    pimc.block_compute(data, results)
    return


@pytest.fixture(params=[(0, 0), (0, 1), (1, 0), (1, 1)])
def dataPM(request):
    data = pimc.BoxDataPM(pibronic.constants.delta_beta)
    data.id_data = request.param[0]
    data.id_rho = request.param[1]
    return data


@pytest.fixture()
def FS_pm(path, dataPM):
    return fs.FileStructure.from_boxdata(path, dataPM)

# this creates 10 different output files for this test case
@pytest.fixture(params=range(0, 10))
def job_id(request):
    return request.param


def test_simple_block_compute_pm(FS_pm, dataPM, job_id):
    samples = int(1e1)
    block_size = int(1e1)
    np.random.seed(242351)  # pick our seed

    dataPM.path_vib_model = FS_pm.path_vib_model
    dataPM.path_rho_model = FS_pm.path_rho_model

    FS_pm.generate_model_hashes()
    dataPM.hash_vib = FS_pm.hash_vib
    dataPM.hash_rho = FS_pm.hash_rho

    # data should be able to figure out the number of states and modes from the file
    dataPM.states = 2
    dataPM.modes = 2

    dataPM.samples = samples
    dataPM.beads = 12
    dataPM.temperature = 300.00
    dataPM.blocks = samples // block_size
    dataPM.block_size = block_size

    # setup empty tensors, models, and constants
    dataPM.preprocess()

    # store results here
    results = pimc.BoxResultPM(data=dataPM)
    results.path_root = FS_pm.path_rho_results
    results.id_job = job_id

    pimc.block_compute_pm(dataPM, results)
    return


# def test_simple_jackknife(FS_pm):
#     pibronic.jackknife.calculate_estimators_and_variance(FS_pm)
#     return
