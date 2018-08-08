""" """

# system imports
import random
import os

# local imports
from .. import context
from pibronic.vibronic import vIO, VMK
from pibronic.data import file_structure as fs

# third party imports
import pytest
from pytest import raises
import numpy as np


@pytest.fixture()
def root(tmpdir_factory):
    return tmpdir_factory.mktemp("root")


@pytest.fixture()
def temp_file_path(root):
    p = root.join("tempfile.json")
    return str(p)


@pytest.fixture()
def temp_FS(root):
    return fs.FileStructure(root, id_data=0, id_rho=0)


@pytest.fixture()
def A():
    return random.randint(2, 10)


@pytest.fixture()
def N():
    return random.randint(2, 10)


@pytest.fixture()
def random_model():
    return vIO.create_random_model()


@pytest.fixture()
def random_diagonal_model():
    return vIO.create_random_diagonal_model()


def test__array_is_symmetric_in_A():
    three_d = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
    assert not vIO._array_is_symmetric_in_A(np.array(three_d))

    three_d = [[[1., 2.], [2., 4.]], [[5., 6.], [6., 8.]]]
    assert vIO._array_is_symmetric_in_A(np.array(three_d))


def test_model_zeros_template_json_dict(A, N):
    model = vIO.model_zeros_template_json_dict(A, N)

    # check for the right type, and that it isn't empty
    assert isinstance(model, dict)
    assert bool(model)

    # check that all elements have the correct value
    assert model[VMK.N] == N
    assert model[VMK.A] == A
    assert np.count_nonzero(model[VMK.E]) is 0
    assert np.count_nonzero(model[VMK.w]) is 0
    assert np.count_nonzero(model[VMK.G1]) is 0
    assert np.count_nonzero(model[VMK.G2]) is 0
    assert np.count_nonzero(model[VMK.G3]) is 0
    assert np.count_nonzero(model[VMK.G4]) is 0


def test__generate_linear_terms(A, N):
    # confirm that the returned array is symmetric in surfaces
    # and that all values are in the correct range
    MAX = 1.0
    shape = vIO.model_shape_dict(A, N)
    displacement = np.random.uniform(0.1, MAX, size=N)
    linear_terms = np.zeros(shape[VMK.G1])

    vIO._generate_linear_terms(linear_terms, shape, displacement, range(N))

    assert np.all(-MAX <= linear_terms) and np.all(linear_terms <= MAX)
    assert vIO._array_is_symmetric_in_A(linear_terms)


def test__generate_quadratic_terms(A, N):
    # confirm that the returned array is symmetric in surfaces and modes
    # and that all values are in the correct range
    MAX = 1.0
    shape = vIO.model_shape_dict(A, N)
    displacement = np.random.uniform(0.1, MAX, size=(N, N))
    quadratic_terms = np.zeros(shape[VMK.G2])

    vIO._generate_quadratic_terms(quadratic_terms, shape, displacement, range(N))

    assert np.all(-MAX <= quadratic_terms) and np.all(quadratic_terms <= MAX)
    assert vIO._array_is_symmetric_in_A(quadratic_terms)


def test_generate_vibronic_model_data(A, N):

    max_E = random.uniform(1.0, 3.0)
    w = random.uniform(0.01, 0.03)
    scaling = random.uniform(0.02, 0.08)

    p = {'frequency_range': [w, 2*w],
         'energy_range': [0.0, max_E],
         'quadratic_scaling': 2*scaling,
         'linear_scaling': scaling,
         'diagonal': False,
         'numStates': A,
         'numModes': N,
         }

    model = vIO.generate_vibronic_model_data(p)
    assert bool(model)


class Test_create_model_hash():

    def test_no_args(self):
        with raises(AssertionError) as e_info:
            vIO.create_model_hash()
        assert str(e_info.value) == "no arguments provided"

    def test_path_only(self, temp_file_path, random_model):
        vIO.save_model_to_JSON(temp_file_path, random_model)
        model_hash = vIO.create_model_hash(path=temp_file_path)

        with open(temp_file_path, mode='r', encoding='UTF8') as file:
            string = file.read()
        assert model_hash == vIO._hash(string)

    def test_FS_only(self, temp_FS, random_model):
        vIO.save_model_to_JSON(temp_FS.path_vib_model, random_model)
        model_hash = vIO.create_model_hash(FS=temp_FS)

        with open(temp_FS.path_vib_model, mode='r', encoding='UTF8') as file:
            string = file.read()
        assert model_hash == vIO._hash(string)


class Test_create_diagonal_model_hash():

    def test_create_diagonal_model_hash_no_args(self):
        with raises(AssertionError) as e_info:
            vIO.create_diagonal_model_hash()
        assert str(e_info.value) == "no arguments provided"

    def test_create_diagonal_model_hash_path_only(self, temp_file_path, random_diagonal_model):
        vIO.save_diagonal_model_to_JSON(temp_file_path, random_diagonal_model)
        diagonal_model_hash = vIO.create_diagonal_model_hash(path=temp_file_path)

        with open(temp_file_path, mode='r', encoding='UTF8') as file:
            string = file.read()
        assert diagonal_model_hash == vIO._hash(string)

    def test_create_diagonal_model_hash_FS_only(self, temp_FS, random_diagonal_model):
        vIO.save_diagonal_model_to_JSON(temp_FS.path_rho_model, random_diagonal_model)
        diagonal_model_hash = vIO.create_diagonal_model_hash(FS=temp_FS)

        with open(temp_FS.path_rho_model, mode='r', encoding='UTF8') as file:
            string = file.read()
        assert diagonal_model_hash == vIO._hash(string)


class Test_extract_dimensions_of_model():

    def test_no_args(self):
        with raises(AssertionError) as e_info:
            vIO.extract_dimensions_of_model()
        assert str(e_info.value) == "no arguments provided"

    def test_path_only(self, temp_file_path, random_model):
        vIO.save_model_to_JSON(temp_file_path, random_model)
        A, N = vIO.extract_dimensions_of_model(path=temp_file_path)
        assert A == random_model[VMK.A]
        assert N == random_model[VMK.N]

    def test_FS_only(self, temp_FS, random_model):
        vIO.save_model_to_JSON(temp_FS.path_vib_model, random_model)
        A, N = vIO.extract_dimensions_of_model(FS=temp_FS)
        assert A == random_model[VMK.A]
        assert N == random_model[VMK.N]


class Test_extract_dimensions_of_diagonal_model():

    def test_no_args(self):
        with raises(AssertionError) as e_info:
            vIO.extract_dimensions_of_diagonal_model()
        assert str(e_info.value) == "no arguments provided"

    def test_path_only(self, temp_file_path, random_diagonal_model):
        vIO.save_diagonal_model_to_JSON(temp_file_path, random_diagonal_model)
        A, N = vIO.extract_dimensions_of_diagonal_model(path=temp_file_path)
        assert A == random_diagonal_model[VMK.A]
        assert N == random_diagonal_model[VMK.N]

    def test_FS_only(self, temp_FS, random_diagonal_model):
        vIO.save_diagonal_model_to_JSON(temp_FS.path_rho_model, random_diagonal_model)
        A, N = vIO.extract_dimensions_of_diagonal_model(FS=temp_FS)
        assert A == random_diagonal_model[VMK.A]
        assert N == random_diagonal_model[VMK.N]


def test__load_from_JSON_no_energy(temp_file_path, random_model):
        random_model[VMK.E].fill(0)
        vIO.save_model_to_JSON(temp_file_path, random_model)
        loaded_model = vIO._load_from_JSON(temp_file_path)
        assert vIO._same_model(random_model, loaded_model)


def test_load_sample_from_JSON_no_arrays_provided(temp_file_path, random_diagonal_model):
        vIO.save_diagonal_model_to_JSON(temp_file_path, random_diagonal_model)
        loaded_d_model = vIO.load_diagonal_model_from_JSON(temp_file_path, dictionary=None)
        assert vIO._same_diagonal_model(random_diagonal_model, loaded_d_model)


def test_create_random_orthonormal_matrix(A):
    mat = vIO.create_random_orthonormal_matrix(A)
    assert np.allclose(mat.dot(mat.T), np.eye(A)), "matrix is not orthonormal"


def test_create_orthonormal_matrix_lambda_close_to_identity(A):
    tuning_parameter = random.uniform(0., 1.0)
    mat = vIO.create_orthonormal_matrix_lambda_close_to_identity(A, tuning_parameter)
    assert np.allclose(mat.dot(mat.T), np.eye(A)), "matrix is not orthonormal"


class Test_create_fake_coupled_model():

    @pytest.fixture()
    def tuning_parameter(self):
        return random.uniform(0., 1.0)

    @pytest.fixture()
    def random_original_model(self, random_model):
        vIO.fill_offdiagonal_of_model_with_zeros(random_model)
        return random_model

    def test_file_no_exist(self, temp_FS, tuning_parameter):
        with raises(AssertionError) as e_info:
            vIO.create_fake_coupled_model(temp_FS, tuning_parameter)
        assert str(e_info.value) == "coupled_model file doesn't exist!"

    def test_basic_file_movement(self, temp_FS, tuning_parameter, random_original_model):
        vIO.save_model_to_JSON(temp_FS.path_vib_model, random_original_model)
        vIO.create_fake_coupled_model(temp_FS, tuning_parameter)

        # check that the original model is saved where we expected
        orig = vIO.load_model_from_JSON(temp_FS.path_orig_model)
        assert vIO._same_model(random_original_model, orig)

        # check that the coupled model is now different
        coupled = vIO.load_model_from_JSON(temp_FS.path_vib_model)
        assert not vIO._same_model(random_original_model, coupled)

        # check that we saved an orthonormal matrix to the appropriate file
        assert os.path.isfile(temp_FS.path_ortho_mat)

    def test_no_U_provided(self, temp_FS, tuning_parameter, random_original_model):
        vIO.save_model_to_JSON(temp_FS.path_vib_model, random_original_model)
        A = random_original_model[VMK.A]
        U_one = vIO.create_orthonormal_matrix_lambda_close_to_identity(A, tuning_parameter)
        vIO.create_fake_coupled_model(temp_FS, tuning_parameter, transformation_matrix=None)
        U_two = np.load(temp_FS.path_ortho_mat)
        assert np.allclose(U_one, U_two)

    def test_U_is_reuse(self, temp_FS, tuning_parameter, random_original_model):
        vIO.save_model_to_JSON(temp_FS.path_vib_model, random_original_model)
        A = random_original_model[VMK.A]
        U_one = vIO.create_orthonormal_matrix_lambda_close_to_identity(A, tuning_parameter)
        np.save(temp_FS.path_ortho_mat, U_one)
        vIO.create_fake_coupled_model(temp_FS, tuning_parameter, transformation_matrix="re-use")
        U_two = np.load(temp_FS.path_ortho_mat)
        assert np.allclose(U_one, U_two)

    def test_U_is_provided(self, temp_FS, tuning_parameter, random_original_model):
        vIO.save_model_to_JSON(temp_FS.path_vib_model, random_original_model)
        A = random_original_model[VMK.A]
        U_one = vIO.create_orthonormal_matrix_lambda_close_to_identity(A, tuning_parameter)
        vIO.create_fake_coupled_model(temp_FS, tuning_parameter, transformation_matrix=U_one)
        U_two = np.load(temp_FS.path_ortho_mat)
        assert np.allclose(U_one, U_two)

    def test_else_case(self, temp_FS, tuning_parameter, random_original_model):
        vIO.save_model_to_JSON(temp_FS.path_vib_model, random_original_model)
        with raises(Exception) as e_info:
            U = "not acceptable input"
            vIO.create_fake_coupled_model(temp_FS, transformation_matrix=U)
        assert str(e_info.value) == "Your transformation_matrix argument is most likely incorrect?"
