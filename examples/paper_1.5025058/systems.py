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
# this dictionary is for conveniently accessing iterators of the coupled model id's for each system
# where the system's name is the key
id_dict = {
    name_lst[0]: list(range(11, 17)),
    name_lst[1]: list(range(21, 27)),
    name_lst[2]: list(range(31, 37)),
    name_lst[3]: list(range(41, 47)),
    }

all_model_ids = [id_dict[key] for key in id_dict.keys()]
id_dict["valid"] = list(it.chain(*all_model_ids))

# this will need to be modified if more rhos/sampling distributions are chosen
# it has no knowledge of the *actual* files in the directory

iterative_model_id_rho = 11  # the id_rho we choose to associate with the iterative model

basic_list = [0, 1]
basic_jahn_teller_list = [0, 1, 2]
iterative_list = basic_list + [iterative_model_id_rho]
iterative_jahn_teller_list = basic_jahn_teller_list + [iterative_model_id_rho]

rho_dict = {
    name_lst[0]: {11: iterative_list,
                  12: iterative_list,
                  13: iterative_list,
                  14: iterative_list,
                  15: iterative_list,
                  16: iterative_list,
                  },
    name_lst[1]: {21: iterative_list,
                  22: iterative_list,
                  23: iterative_list,
                  24: iterative_list,
                  25: iterative_list,
                  26: iterative_list,
                  },
    name_lst[2]: {31: iterative_list,
                  32: iterative_list,
                  33: iterative_list,
                  34: iterative_list,
                  35: iterative_list,
                  36: iterative_list,
                  },
    name_lst[3]: {41: iterative_jahn_teller_list,
                  42: iterative_jahn_teller_list,
                  43: iterative_jahn_teller_list,
                  44: iterative_jahn_teller_list,
                  45: iterative_jahn_teller_list,
                  46: iterative_jahn_teller_list,
                  },
}

"""
the old rho 2,3 files present in /alternate_rhos/ (previously present in commit (804d90ce68e3ba3275724d4cd01bd19cc5f0afc8)) for the [superimposed, displaced, elevated] models are old examples of systems that might be worth investigating, with proper re-calculation of the weights of the oscillators based on single-point energy calculations than the "additional" oscillators shouldn't have any significant affect on the sampling

I believe the old jahn-teller rho 3's are ones where the zero of energy has been removed?
if we come back to these models it would be worth confirming this.
"""


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
