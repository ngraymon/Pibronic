""" fixed data for each of the four systems """

# system imports
import itertools as it

# local imports

# third party imports


# each system has a name
# the names are used to identify the input files in the directories
# input_mctdh_files, input_json, alternate_rhos
name_lst = ["superimposed", "displaced", "elevated", "jahnteller"]

# each system is examined for 6 different values of a specific parameter
# thus each system is associated with 6 models
# this dictionary is for conviently accesing iterators of the coupled model id's for each system
# where the system's name is the key
id_dict = {
    "superimposed": list(range(11, 17)),
    "displaced":    list(range(21, 27)),
    "elevated":     list(range(31, 37)),
    "jahnteller":   list(range(41, 47)),
    "valid": list(it.chain(
                    range(11, 17),
                    range(21, 27),
                    range(31, 37),
                    range(41, 47),
                    )),
    }

# all the rho files present in /alternate_rhos/
# this would need to be modified if more rhos/sampling distributions where chosen
# it has no knowledge of the *actual* files in the directory
rho_dict = {
    "superimposed": {11: range(3),
                     12: range(3),
                     13: range(3),
                     14: range(3),
                     15: range(4),
                     16: range(4),
                     },
    "displaced":    {21: range(4),
                     22: range(3),
                     23: range(3),
                     24: range(3),
                     25: range(4),
                     26: range(4),
                     },
    "elevated":     {31: range(2),
                     32: range(2),
                     33: range(2),
                     34: range(2),
                     35: range(3),
                     36: range(3),
                     },
    "jahnteller":   {41: range(3),
                     42: range(3),
                     43: range(3),
                     44: range(3),
                     45: range(4),
                     46: range(4),
                     },
}


def id_data_is_valid(id_data):
    """returns bool
    tests if id_data is a member of the list corresponding to the key "valid" in the id_dict"""
    return id_data in id_dict["valid"]


def assert_id_data_is_valid(id_data):
    """ assert wrapper around id_data_is_valid() """
    assert id_data_is_valid(id_data), f"Invalid id_data ({id_data:d})"
    return


def system_name_is_valid(name):
    """returns bool
    tests if name is a member of the list name_lst"""
    return name in name_lst


def assert_system_name_is_valid(name):
    """ assert wrapper around system_name_is_valid() """
    assert system_name_is_valid(name), f"Invalid system name ({name:s})"
    return


def get_system_name(id_data):
    """ each system name is associated with 6 different data id's
    they are identified by the number in the ten's position """
    assert_id_data_is_valid(id_data)

    n = id_data // 10

    d = {
         1: "superimposed",
         2: "displaced",
         3: "elevated",
         4: "jahnteller",
         }

    name = d[n]

    return name
