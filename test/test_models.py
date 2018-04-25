"""A basic test of the minimal module
"""

# system imports
import unittest

# third party imports
import numpy as np
import pibronic
import pibronic.data.file_structure as fs
import pibronic.pimc.minimal as pm

# local imports


class TestDataSet0Rho0(unittest.TestCase):
    def setUp(self):
        self.id_data = 0
        self.id_rho = 0

        np.random.seed(242351)  # pick our seed
        self.samples = int(1e1)
        self.block_size = int(1e1)
        self.test_path = '/home/ngraymon/test/Pibronic/test/test_models/'

    def test_load(self):
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


class TestDataSet1Rho0(unittest.TestCase):
    def setUp(self):
        self.id_data = 1
        self.id_rho = 0

        np.random.seed(242351)  # pick our seed
        self.samples = int(1e1)
        self.block_size = int(1e1)
        self.test_path = '/home/ngraymon/test/Pibronic/test/test_models/'

    def test_load(self):
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


if __name__ == '__main__':
    unittest.main()
