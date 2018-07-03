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
    "2x2":   [0],
    "4x6":   [1],
    "7x12":  [2],
    }

all_model_ids = [id_dict[key] for key in id_dict.keys()]
id_dict["valid"] = list(it.chain(*all_model_ids))

# all the rho files present in /alternate_rhos/
# this would need to be modified if more rhos/sampling distributions where chosen
# it has no knowledge of the *actual* files in the directory
rho_dict = {
    "2x2":   {0: [*range(2)]},
    "4x6":   {1: [*range(2)]},
    "7x12":  {2: [*range(2)]},
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
