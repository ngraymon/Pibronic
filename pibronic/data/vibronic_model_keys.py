""" contains the VibronicModelKeys """

from enum import Enum


class VibronicModelKeys(Enum):
    """The VibronicModelKeys, which are the keys (strings) used in the .json files to identify the corresponding values
    """
    number_of_modes = "number of modes"
    number_of_surfaces = "number of surfaces"
    energies = "energies"
    frequencies = "frequencies"
    linear_couplings = "linear couplings"
    quadratic_couplings = "quadratic couplings"
    cubic_couplings = "cubic couplings"
    quartic_couplings = "quartic couplings"

    # aliases for the enum members
    N = number_of_modes
    A = number_of_surfaces
    E = energies
    w = frequencies
    G1 = linear_couplings
    G2 = quadratic_couplings
    G3 = cubic_couplings
    G4 = quartic_couplings

    @classmethod
    def change_dictionary_keys_from_enum_members_to_strings(cls, input_dict):
        """ does what it says """
        for key, value in list(input_dict.items()):
            if key in cls:
                input_dict[key.value] = value
                del input_dict[key]
        return

    @classmethod
    def change_dictionary_keys_from_strings_to_enum_members(cls, input_dict):
        """ does what it says """
        for key, value in list(input_dict.items()):
            input_dict[cls(key)] = value
            del input_dict[key]
        return

    @classmethod
    def key_list(cls):
        """ returns a list of all enum members that are ommited from the .json file if all of their array's values are 0
        """
        return [cls.E, cls.G1, cls.G2, cls.G3, cls.G4]
