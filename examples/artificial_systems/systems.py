""" fixed data for each of the artificial systems """

# system imports
import itertools as it

# local imports

# third party imports

# each system has a name
# the names are used to identify the input files in the directories
# as well as provide an easy way to distinguish them
name_lst = ["2x2", "4x6", "7x12"]

# this dictionary is for conveniently accessing iterators of the coupled model id's for each system
# where the system's name is the key
id_dict = {
    name_lst[0]: list(range(11, 16)),
    name_lst[1]: list(range(21, 26)),
    name_lst[2]: list(range(31, 36)),
    }

all_model_ids = [id_dict[key] for key in id_dict.keys()]
id_dict["valid"] = list(it.chain(*all_model_ids))

# all the rho files present in /alternate_rhos/
# this would need to be modified if more rhos/sampling distributions where chosen
# it has no knowledge of the *actual* files in the directory
num_of_rhos = 2

rho_dict = {
    name_lst[0]: {11: list(range(num_of_rhos)),
                  12: list(range(num_of_rhos)),
                  13: list(range(num_of_rhos)),
                  14: list(range(num_of_rhos)),
                  15: list(range(num_of_rhos)),
                  16: list(range(num_of_rhos)),
                  },
    name_lst[1]: {21: list(range(num_of_rhos)),
                  22: list(range(num_of_rhos)),
                  23: list(range(num_of_rhos)),
                  24: list(range(num_of_rhos)),
                  25: list(range(num_of_rhos)),
                  },
    name_lst[2]: {31: list(range(num_of_rhos)),
                  32: list(range(num_of_rhos)),
                  33: list(range(num_of_rhos)),
                  34: list(range(num_of_rhos)),
                  35: list(range(num_of_rhos)),
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
