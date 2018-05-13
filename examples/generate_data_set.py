# minimal.py

# system imports
import os
from os.path import join
import shutil

# third party imports
# import numpy as np

# local imports
from context import pibronic
import pibronic.data.file_structure as fs
import pibronic.data.vibronic_model_io as vIO
# import pibronic.data.file_name as fn


# you can change this
root = os.getcwd()
# create the FileStructure object which will create the directories need to organize the files
FS = fs.FileStructure(root, 0, 0)

# copy the provided fake_vibron file into the electronic structure directory
new_path = shutil.copy(join(root, "fake_vibron.op"), FS.path_es)

# create the coupled_model.json file from the fake_vibron.op file
# this is not really necessary given that this is a test case
# but we want to be able to replace the fake_vibron.op data with real data eventually
vIO.create_coupling_from_op_file(FS, new_path)
vIO.create_harmonic_model(FS)
vIO.create_basic_sampling_model(FS)

