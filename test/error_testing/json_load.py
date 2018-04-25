# quick reading of json files

from .context import pibronic
import numpy as np
import json
import sys
import os


import importlib.machinery
vIO_dir = "/home/ngraymon/thesis_code/pimc/" if sys.platform.startswith('linux') else "/Users/ngraymon/Documents/masters/thesis/code/pimc/"
loader = importlib.machinery.SourceFileLoader("vIO", os.path.abspath(vIO_dir + "vibronic_model_io.py"))
vIO = loader.load_module("vIO")


def jload(filename):
    return json.load(open(filename, mode='r', encoding='UTF8'))
