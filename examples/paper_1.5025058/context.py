import socket
import sys
import os

# import the path to the pibronic package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pibronic


# these parameters should be modified to fit your use case
# -----------------------------------------------------------------------------------------------
# list of possible server hostnames for which we will store results at /default_server_root/
list_of_server_hostnames = ["nlogn", "feynman"]

# the pre-defined root when executing on your server
default_server_root = "/work/ngraymon/pimc/paper_2018/"


def choose_root_folder():
    """ just a wrapper for choosing the root folder where data will be stored
    easily changed to suit your use case"""

    # (default)  creates the output files in the /Pibronic/examples/ folder
    root = os.path.abspath(os.path.dirname(__file__))

    hostname = socket.gethostname()

    # if we are on a server then we want to use a different location
    if hostname in list_of_server_hostnames:
        root = default_server_root
    else:
        raise Exception(f"Can't handle hostname ({hostname:})")

    return root

# -----------------------------------------------------------------------------------------------
