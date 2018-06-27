# generate_plots.py - creates all the directories and copies in the necessary files

# system imports
import socket
import shutil
import glob
import os
from os.path import join

# third party imports

# local imports
from context import pibronic
from pibronic.vibronic import vIO
import pibronic.data.file_structure as fs

# list of the names for each of the four system's
from context import system_names

# lists of the coupled model id's for each system
# in a dictionary, with the system's name as the key
from context import data_dict


def automate_simple_z_plots():
    return


if (__name__ == "__main__"):

    automate_simple_z_plots()
