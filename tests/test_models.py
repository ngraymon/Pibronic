"""A basic test of the minimal module
"""

# system imports
import os

# local imports
from .context import pibronic
import pibronic.data.vibronic_model_io as vIO
import pibronic.data.file_structure as fs
import pibronic.pimc.minimal as minimal
import pibronic.constants

# third party imports
import filecmp
import pytest
import numpy as np


class TestProcessingVibronicModel():

    test_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_models/")

    # check both the data set 0 and data set 1
    @pytest.fixture(params=[(0, 0), (0, 1), (1, 0), (1, 1)])
    def files(self, request):
        return fs.FileStructure(self.test_path, request.param[0], request.param[1])

    # looks like we might not use this file type anymore
    # def test_creating_coupled_model_from_h_file(self, files):
    #     path_h_file = files.path_es + "fake_vibron.h"
    #     vIO.create_coupling_from_h_file(files, path_h_file)
    #     samefile = filecmp.cmp(files.path_vib_model, files.path_vib_params + "coupled_model_reference.json")
    #     assert samefile
    #     return

    def test_creating_harmonic_model_from_coupled_model(self, files):
        vIO.create_harmonic_model(files)
        samefile = filecmp.cmp(files.path_har_model, files.path_vib_params + "harmonic_model_reference.json")
        assert samefile
        return

    # only check the data_set_0
    @pytest.fixture(params=[(0, 0), (0, 1)])
    def files(self, request):
        return fs.FileStructure(self.test_path, request.param[0], request.param[1])

    # only want to check that the rho 0 sampling models are the same as the harmonic models
    def test_creating_basic_sampling_model_from_harmonic_model(self, files):
        vIO.create_basic_sampling_model(files)
        samefile = filecmp.cmp(files.path_rho_model, files.path_rho_params + "sampling_model_reference.json")
        assert samefile
        return

    # seems to work now
    # # @pytest.fixture(params=[(0, 0), (0, 1), (1, 0), (1, 1)])
    # @pytest.fixture(params=[(0, 0)])
    # def files(self, request):
    #     return fs.FileStructure(self.test_path, request.param[0], request.param[1])

    # def test_create_coupling_from_op_file(self, files):
    #     path_full = files.path_es + "fake_vibron.op"
    #     vIO.create_coupling_from_op_file(files, path_full)
    #     assert os.path.isfile(files.path_vib_model), "failed to create any output file"
    #     assert True, "output values are incorrect"  # TODO - implement way to check all the values of the output
    #     return

    # # check both the data set 0 and data set 1
    # # @pytest.fixture(params=[(0, 0), (0, 1), (1, 0), (1, 1)])
    # @pytest.fixture(params=["ch3cn", "acrolein", "acrylonitrile", ])
    # def https(self, request):
    #     root = "http://scienide2.uwaterloo.ca/~nooijen/website_new_20_10_2011/vibron/VC/"
    #     url = root + request.param + "_files.html"
    #     return url

    # import urllib

    # def test_create_coupling_from_op_hyperlink(self, files, https):
    #     vIO.create_coupling_from_op_hyperlink(files, https)
    #     assert os.path.isfile(files.path_vib_model), "failed to create any output file"
    #     assert True, "output values are incorrect"  # TODO - implement way to check all the values of the output
    #     # TODO
    #     return


class TestMinimalNatively():
    samples = int(1e1)
    block_size = int(1e1)
    np.random.seed(242351)  # pick our seed

    test_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_models/")

    def test_create_BoxData_inline(self):
        data = minimal.BoxData()
        return

    def test_create_BoxData_from_file(self):
        return

    def test_create_BoxData_from_jsonData(self):
        return

    @pytest.fixture(scope="class", params=[(0, 0), (0, 1), (1, 0), (1, 1)])
    def data(self, request):
        data = minimal.BoxData()
        data.id_data = request.param[0]
        data.id_rho = request.param[1]
        return data

    @pytest.fixture(scope="class")
    def files(self, data):
        return fs.FileStructure.from_boxdata(self.test_path, data)

    def test_simple_block_compute(self, files, data):
        data.path_vib_model = files.path_vib_model
        data.path_rho_model = files.path_rho_model

        data.states = 2
        data.modes = 2

        data.samples = self.samples
        data.beads = 12
        data.temperature = 300.0
        data.blocks = self.samples // self.block_size
        data.block_size = self.block_size

        # setup empty tensors, models, and constants
        data.preprocess()

        # store results here
        results = minimal.BoxResult(data=data)
        results.path_root = files.path_rho_results

        minimal.block_compute(data, results)
        return

    @pytest.fixture(scope="class", params=[(0, 0), (0, 1), (1, 0), (1, 1)])
    def dataPM(self, request):
        data = minimal.BoxDataPM(pibronic.constants.delta_beta)
        data.id_data = request.param[0]
        data.id_rho = request.param[1]
        return data

    @pytest.fixture(scope="class")
    def filesPM(self, dataPM):
        return fs.FileStructure.from_boxdata(self.test_path, dataPM)

    # this creates 10 different output files for this test case
    @pytest.fixture(scope="class", params=range(0, 10))
    def job_id(self, request):
        return request.param

    def test_simple_block_compute_pm(self, filesPM, dataPM, job_id):
        dataPM.path_vib_model = filesPM.path_vib_model
        dataPM.path_rho_model = filesPM.path_rho_model

        # data should be able to figure out the number of states and modes from the file
        dataPM.states = 2
        dataPM.modes = 2

        dataPM.samples = self.samples
        dataPM.beads = 12
        dataPM.temperature = 300.00
        dataPM.blocks = self.samples // self.block_size
        dataPM.block_size = self.block_size

        # setup empty tensors, models, and constants
        dataPM.preprocess()

        # store results here
        results = minimal.BoxResultPM(data=dataPM)
        results.path_root = filesPM.path_rho_results
        results.id_job = job_id

        minimal.block_compute_pm(dataPM, results)
        return
