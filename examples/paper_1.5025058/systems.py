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
    name_lst[0]: list(range(11, 17)),
    name_lst[1]: list(range(21, 27)),
    name_lst[2]: list(range(31, 37)),
    name_lst[3]: list(range(41, 47)),
    }

all_model_ids = [id_dict[key] for key in id_dict.keys()]
id_dict["valid"] = list(it.chain(*all_model_ids))

# all the rho files present in /alternate_rhos/
# this would need to be modified if more rhos/sampling distributions where chosen
# it has no knowledge of the *actual* files in the directory
rho_dict = {
    name_lst[0]: {11: list(range(3)),
                  12: list(range(3)),
                  13: list(range(3)),
                  14: list(range(3)),
                  15: list(range(4)),
                  16: list(range(4)),
                  },
    name_lst[1]: {21: list(range(4)),
                  22: list(range(3)),
                  23: list(range(3)),
                  24: list(range(3)),
                  25: list(range(4)),
                  26: list(range(4)),
                  },
    name_lst[2]: {31: list(range(2)),
                  32: list(range(2)),
                  33: list(range(2)),
                  34: list(range(2)),
                  35: list(range(3)),
                  36: list(range(3)),
                  },
    name_lst[3]: {41: list(range(3)),
                  42: list(range(3)),
                  43: list(range(3)),
                  44: list(range(3)),
                  45: list(range(4)),
                  46: list(range(4)),
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

    for name in name_lst:
        if id_data in id_dict[name]:
            return name
    else:
        raise Exception("This shouldn't happen")
