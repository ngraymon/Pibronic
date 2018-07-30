""" An explicit comparison of values for a very small sample set with fixed outputs
    This should demonstrate that the results of the code are consistent across commits
"""

# system imports
import shutil
import os
from os.path import dirname, join, abspath

# third party imports
import pytest
import numpy as np
# from numpy import float64 as F64

# local imports
from .. import context
from pibronic import pimc
from pibronic import constants
import pibronic.data.file_structure as fs
from pibronic.vibronic import vIO, VMK


RTOL = 1e-05
ATOL = 1e-08


@pytest.fixture(scope="module")
def temperature_fixture():
    temperature = 300.00
    return temperature


@pytest.fixture(scope="module")
def beta_fixture(temperature_fixture):
    B = constants.beta(temperature_fixture)
    assert np.allclose(B, 38.68174020133669, rtol=RTOL, atol=ATOL), f"Incorrect beta value\n{B:}"
    return B


@pytest.fixture(scope="module")
def path_src():
    return join(dirname(abspath(__file__)), "explicit_data")


@pytest.fixture(scope="module")
def root(tmpdir_factory):
    return tmpdir_factory.mktemp("root")


@pytest.fixture(scope="module")
def FS(root, path_src):
    FS = fs.FileStructure(root, id_data=0, id_rho=0)
    path_model_src = join(path_src, "coupled_model.json")
    shutil.copy(path_model_src, FS.path_vib_model)  # place the coupled model in the directory
    path_sampling_src = join(path_src, "sampling_model.json")
    shutil.copy(path_sampling_src, FS.path_rho_model)  # place the sampling model in the directory
    return FS


@pytest.fixture(scope="module")
def data(FS):
    data = pimc.BoxData.from_FileStructure(FS)

    err_str = "data didn't build correctly"
    assert data.id_data is 0, err_str
    assert data.id_rho is 0, err_str
    assert data.path_vib_model == FS.path_vib_model, err_str
    assert data.path_rho_model == FS.path_rho_model, err_str

    model = vIO.load_model_from_JSON(FS.path_vib_model)
    assert data.states is model[VMK.number_of_surfaces], err_str
    assert data.modes is model[VMK.number_of_modes], err_str

    n_samples = 10
    block_size = 2  # we will do 5 blocks of 2 samples for a total of 10 samples

    data.samples = n_samples
    data.beads = 5
    data.temperature = temperature_fixture()
    data.blocks = data.samples // block_size
    data.block_size = block_size

    data.preprocess()
    return data


def test_surface_weights(data, path_src):
    # c_surface_weights = np.load(join(path_src, "coupled_surface_weights.npy"))
    s_surface_weights = np.load(join(path_src, "sampling_surface_weights.npy"))

    # these weights aren't normalized
    # c_surface_weights /= c_surface_weights.sum()
    s_surface_weights /= s_surface_weights.sum()

    # we don't compare the coupled_surface_weights at the moment as they never get used
    # assert np.allclose(c_surface_weights, data.vib.state_weight, rtol=RTOL, atol=ATOL)
    assert np.allclose(s_surface_weights, data.rho.state_weight, rtol=RTOL, atol=ATOL)
    return


def test_multivariate_normal_distributions(data, path_src):
    c_means = np.load(join(path_src, "coupled_means.npy"))
    s_means = np.load(join(path_src, "sampling_means.npy"))
    # c_covariance = np.load(join(path_src, "coupled_covariance.npy"))
    s_covariance = np.load(join(path_src, "sampling_covariance.npy"))

    # make sure they are identical in the bead dimension
    for idx in range(c_means.shape[-1] - 1):
        assert np.allclose(c_means[:, :, idx], c_means[:, :, idx+1], rtol=RTOL, atol=ATOL)
        assert np.allclose(s_means[:, :, idx], s_means[:, :, idx+1], rtol=RTOL, atol=ATOL)

    assert np.allclose(c_means[..., 0], data.vib.state_shift, rtol=RTOL, atol=ATOL)
    assert np.allclose(s_means[..., 0], data.rho.state_shift, rtol=RTOL, atol=ATOL)

    # build the un-diagonalized covariance matrix
    from numpy import newaxis as NEW
    left = 2. * data.rho.const.cothAN[..., NEW, NEW] * np.eye(5)
    right = data.rho.const.cschAN[..., NEW, NEW] * data.circulant_matrix[NEW, NEW, ...]
    inverse_covariance = left - right
    cov = np.linalg.inv(inverse_covariance)

    assert np.allclose(s_covariance, cov, rtol=RTOL, atol=ATOL)
    return


def test_energy_offsets_due_to_linear_terms(data, path_src):
    c_Edeltas = np.load(join(path_src, "coupled_Edeltas.npy"))
    s_Edeltas = np.load(join(path_src, "sampling_Edeltas.npy"))
    assert np.allclose(c_Edeltas, data.vib.delta_weight, rtol=RTOL, atol=ATOL)
    assert np.allclose(s_Edeltas, data.rho.delta_weight, rtol=RTOL, atol=ATOL)
    return


def test_position_offsets_due_to_linear_terms(data, path_src):
    c_ds = np.load(join(path_src, "coupled_ds.npy"))
    s_ds = np.load(join(path_src, "sampling_ds.npy"))
    assert np.allclose(c_ds, data.vib.state_shift, rtol=RTOL, atol=ATOL)
    assert np.allclose(s_ds, data.rho.state_shift, rtol=RTOL, atol=ATOL)
    return


def test_precomputed_coth(data, path_src):
    c_cosh = np.load(join(path_src, "coupled_cosh.npy"))
    c_sinh = np.load(join(path_src, "coupled_sinh.npy"))
    s_cosh = np.load(join(path_src, "sampling_cosh.npy"))
    s_sinh = np.load(join(path_src, "sampling_sinh.npy"))
    assert np.allclose(c_cosh/c_sinh, data.vib.const.cothAN, rtol=RTOL, atol=ATOL)
    assert np.allclose(s_cosh/s_sinh, data.rho.const.cothAN, rtol=RTOL, atol=ATOL)
    return


def test_precomputed_csch(data, path_src):
    c_sinh = np.load(join(path_src, "coupled_sinh.npy"))
    s_sinh = np.load(join(path_src, "sampling_sinh.npy"))
    assert np.allclose(c_sinh**-1., data.vib.const.cschAN, rtol=RTOL, atol=ATOL)
    assert np.allclose(s_sinh**-1., data.rho.const.cschAN, rtol=RTOL, atol=ATOL)
    return


@pytest.fixture(scope="module")
def results(FS, data):
    results = pimc.BoxResult(data=data)
    results.path_root = FS.path_rho_results
    results.id_job = 0
    return results


def test_block_compute(FS, data, results, path_src):
    from pibronic.pimc.pimc import build_o_matrix, build_denominator, diagonalize_coupling_matrix, build_numerator
    # labels for clarity
    rho = data.rho
    vib = data.vib

    # store results here
    y_rho = results.scaled_rho.view()
    y_g = results.scaled_g.view()

    samples = np.load(join(path_src, "samples.npy"))

    rho_oMatricies = np.load(join(path_src, "rho_oMat.npy"))
    vib_oMatricies = np.load(join(path_src, "vib_oMat.npy"))
    vib_mMatricies = np.load(join(path_src, "vib_mMat.npy"))
    numerator = np.load(join(path_src, "numerator(g).npy"))
    denominator = np.load(join(path_src, "denominator(rho).npy"))

    for block_index in range(0, data.blocks):
        # indices
        start = block_index * data.block_size
        end = (block_index + 1) * data.block_size
        sample_view = slice(start, end)

        data.qTensor = np.zeros(data.size['BANP'])

        data.qTensor = samples[sample_view, ...]

        # build O matrices for sampling distribution
        build_o_matrix(data, rho.const, rho.state_shift)
        assert np.allclose(rho.const.omatrix,
                           rho_oMatricies[sample_view, ...], rtol=RTOL, atol=ATOL)
        build_o_matrix(data, vib.const, vib.state_shift)
        assert np.allclose(vib.const.omatrix,
                           vib_oMatricies[sample_view, ...], rtol=RTOL, atol=ATOL)

        build_denominator(rho.const, y_rho, sample_view)
        diagonalize_coupling_matrix(data)
        build_numerator(data, vib.const, y_g, sample_view)
        assert np.allclose(data.M_matrix, vib_mMatricies[sample_view, ...], rtol=RTOL, atol=ATOL)

    assert np.allclose(y_rho, denominator, rtol=RTOL, atol=ATOL)
    assert np.allclose(y_g, numerator, rtol=RTOL, atol=ATOL)
    return
