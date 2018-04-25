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

        np.random.seed(232942)  # pick our seed
        self.samples = int(1e2)
        self.block_size = int(1e2)
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
        data.beads = 1000
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

# def data_set_0_test(id_data, id_rho=0):
#     """Just do simple expval(Z) calculation"""
#     np.random.seed(232942)  # pick our seed
#     samples = int(1e2)
#     Bsize = int(1e2)

#     # load the relevant data
#     data = BoxData()
#     data.id_data = id_data
#     data.id_rho = id_rho

#     files = file_structure.FileStructure('/work/ngraymon/pimc/', id_data, id_rho)
#     data.path_vib_model = files.path_vib_model
#     data.path_rho_model = files.path_rho_model

#     data.states = 2
#     data.modes = 2

#     data.samples = samples
#     data.beads = 1000
#     data.temperature = 300.0
#     data.blocks = samples // Bsize
#     data.block_size = Bsize

#     # setup empty tensors, models, and constants
#     data.preprocess()

#     # store results here
#     results = BoxResult(data=data)
#     results.path_root = files.path_rho_results

#     block_compute(data, results)
#     return


if __name__ == '__main__':
    unittest.main()