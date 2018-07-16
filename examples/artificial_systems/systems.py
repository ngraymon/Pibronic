""" fixed data for each of the artifical systems """

# system imports
import itertools as it

# local imports

# third party imports


# each system has a name
# the names are used to identify the input files in the directories
# as well as provide an easy way to distinguish them
name_lst = ["2x2", "4x6", "7x12"]

# this dictionary is for conviently accesing iterators of the coupled model id's for each system
# where the system's name is the key
id_dict = {
    name_lst[0]: [11, 12, 13, 14, 15],
    name_lst[1]: [21, 22, 23, 24, 25],
    name_lst[2]: [31, 32, 33, 34, 35],
    }

all_model_ids = [id_dict[key] for key in id_dict.keys()]
id_dict["valid"] = list(it.chain(*all_model_ids))

# all the rho files present in /alternate_rhos/
# this would need to be modified if more rhos/sampling distributions where chosen
# it has no knowledge of the *actual* files in the directory
rho_dict = {
    name_lst[0]: {11: list(range(2)),
                  12: list(range(2)),
                  13: list(range(2)),
                  14: list(range(2)),
                  15: list(range(2)),
                  16: list(range(2)),
                  },
    name_lst[1]: {21: list(range(2)),
                  22: list(range(2)),
                  23: list(range(2)),
                  24: list(range(2)),
                  25: list(range(2)),
                  },
    name_lst[2]: {31: list(range(2)),
                  32: list(range(2)),
                  33: list(range(2)),
                  34: list(range(2)),
                  35: list(range(2)),
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


def get_tuning_parameter(id_data):
    """ each system name is associated with 6 different data id's
    they are identified by the number in the ten's position """
    assert_id_data_is_valid(id_data)

    ones = id_data % 10

    # these are the transformation matrix tuning parameter values associated with the models
    param_dict = {1: 0.05,
                  2: 0.25,
                  3: 0.50,
                  4: 0.75,
                  5: 1.0,
                  }

    return param_dict[ones]


def get_system_name(id_data):
    """ each system name is associated with 5 different data id's
    they are identified by the number in the ten's position """
    assert_id_data_is_valid(id_data)

    for name in name_lst:
        if id_data in id_dict[name]:
            return name
    else:
        raise Exception("This shouldn't happen")
