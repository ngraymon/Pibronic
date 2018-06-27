import socket
import sys
import os

# import the path to the pibronic package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pibronic

# these parameters should be modified to fit your use case
# -----------------------------------------------------------------------------------------------
# list of possible server hostnames for which we will store results at /server_root/
list_of_server_hostnames = ["nlogn", "feynman"]

# the pre-defined root when executing on your server
server_root = "/work/ngraymon/pimc/paper_2018/"


def choose_root_folder():
    """ just a wrapper for choosing the root folder where data will be stored
    easily changed to suit your use case"""

    # (default)  creates the output files in the /Pibronic/examples/ folder
    root = os.path.abspath(os.path.dirname(__file__))

    # if we are on a serve then we want to use a different location
    if socket.gethostname() in list_of_server_hostnames:
        root = server_root
    else:
        print("Can't handle hostname {:}".format(socket.gethostname()))

    return root
# -----------------------------------------------------------------------------------------------


# these parameters don't need to be modified
# -----------------------------------------------------------------------------------------------
# list of the names for each of the four system's
system_names = ["superimposed", "displaced", "elevated", "jahnteller"]

# lists of the coupled model id's for each system
# in a dictionary, with the system's name as the key
data_dict = {
    "superimposed": range(11, 17),
    "displaced":    range(21, 27),
    "elevated":     range(31, 37),
    "jahnteller":   range(41, 47),
    }


def get_system_name(id_data):
    n = id_data // 10

    d = {
         1: "superimposed",
         2: "displaced",
         3: "elevated",
         4: "jahnteller",
         }

    name = d[n]

    return name

# -----------------------------------------------------------------------------------------------