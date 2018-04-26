"""A basic test of the minimal module
"""

# system imports
# import unittest
import pytest
import filecmp

# third party imports
import numpy as np
import pibronic.data.file_structure as fs
import pibronic.data.vibronic_model_io as vIO
import pibronic.pimc.minimal as pm

# local imports

# unittests can have subTests - which is pretty cool
# def test_should_all_be_even(self):
#     for n in (0, 4, -2, 11):
#         with self.subTest(n=n):
#             self.assertTrue(is_even(n))

# class TestDataSet0Rho0(unittest.TestCase):
#     def setUp(self):
#         self.test_path = '/home/ngraymon/test/Pibronic/test/test_models/'
#         self.molecule_name = "fake"
#         self.id_data = 0
#         self.id_rho = 0
#         return


# class TestDataSet0Rho1(unittest.TestCase):
#     def setUp(self):
#         self.test_path = '/home/ngraymon/test/Pibronic/test/test_models/'
#         self.molecule_name = "fake"
#         self.id_data = 0
#         self.id_rho = 1
#         return


# class TestDataSet1Rho0(unittest.TestCase):
#     def setUp(self):
#         self.test_path = '/home/ngraymon/test/Pibronic/test/test_models/'
#         self.molecule_name = "fake"
#         self.id_data = 1
#         self.id_rho = 0
#         return


# class TestDataSet1Rho1(unittest.TestCase):
#     def setUp(self):
#         self.test_path = '/home/ngraymon/test/Pibronic/test/test_models/'
#         self.molecule_name = "fake"
#         self.id_data = 1
#         self.id_rho = 1
#         return


class TestProcessingVibronicModel():

    test_path = '/home/ngraymon/test/Pibronic/test/test_models/'

    @pytest.fixture(scope="class", params=[(0, 0), (0, 1), (1, 0), (1, 1)])
    def files(self, request):
        return fs.FileStructure(self.test_path, request.param[0], request.param[1])

    def test_creating_coupled_model_from_h_file(self, files):
        path_h_file = files.path_es + "fake_vibron.h"
        vIO.create_coupling_from_h_file(path_h_file)
        samefile = filecmp.cmp(files.path_vib_model, files.path_vib_params + "coupled_model_reference.json")
        assert samefile
        return

    def test_creating_harmonic_model_from_coupled_model(self, files):
        vIO.create_harmonic_model(files)
        samefile = filecmp.cmp(files.path_har_model, files.path_vib_params + "harmonic_model_reference.json")
        assert samefile
        return

    # only want to check that the rho 0 sampling models are the same as the harmonic models
    # def test_creating_basic_sampling_model_from_harmonic_model(files):
    #     vIO.create_basic_sampling_model(files)
    #     samefile = filecmp.cmp(files.path_rho_model, files.path_rho_params + "sampling_model_reference.json")
    #     assert samefile
    #     return


class TestMinimal():
    samples = int(1e1)
    block_size = int(1e1)
    np.random.seed(242351)  # pick our seed

    test_path = '/home/ngraymon/test/Pibronic/test/test_models/'

    @pytest.fixture(scope="class", params=[(0, 0), (0, 1), (1, 0), (1, 1)])
    def data(self, request):
        data = pm.BoxData()
        data.id_data = request.param[0]
        data.id_rho = request.param[1]
        return data

    def test_load(self, data):
        # should be able to pass a data object to file_structure
        files = fs.FileStructure.from_boxdata(self.test_path, data)
        data.path_vib_model = files.path_vib_model
        data.path_rho_model = files.path_rho_model

        data.states = 2
        data.modes = 2

        data.samples = self.samples
        data.beads = 10
        data.temperature = 300.0
        data.blocks = self.samples // self.block_size
        data.block_size = self.block_size

        # setup empty tensors, models, and constants
        data.preprocess()

        # store results here
        results = pm.BoxResult(data=data)
        results.path_root = files.path_rho_results

        pm.block_compute(data, results)

        return


class TestDataSet1Rho0():
    def test_load(self):
        self.id_data = 1
        self.id_rho = 0

        np.random.seed(242351)  # pick our seed
        self.samples = int(1e1)
        self.block_size = int(1e1)
        self.test_path = '/home/ngraymon/test/Pibronic/test/test_models/'

        # load the relevant data
        data = pm.BoxData()
        data.id_data = self.id_data
        data.id_rho = self.id_rho

        # should be able to pass a data object to file_structure
        files = fs.FileStructure.from_boxdata(self.test_path, data)
        data.path_vib_model = files.path_vib_model
        data.path_rho_model = files.path_rho_model

        data.states = 2
        data.modes = 2

        data.samples = self.samples
        data.beads = 10
        data.temperature = 300.0
        data.blocks = self.samples // self.block_size
        data.block_size = self.block_size

        # setup empty tensors, models, and constants
        data.preprocess()

        # store results here
        results = pm.BoxResult(data=data)
        results.path_root = files.path_rho_results

        pm.block_compute(data, results)

        return


# if __name__ == '__main__':
#     runner = unittest.TextTestRunner()
#     runner.run(suite())
