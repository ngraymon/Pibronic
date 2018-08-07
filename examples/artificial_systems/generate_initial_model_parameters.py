# run this script to regenerate the initial model parameters for larger systems

# system imports
import os
from os.path import join, abspath, dirname

# third party imports

# local imports
import context
import systems
from pibronic.vibronic import vIO


parameters = {

    # 4x6
    systems.name_lst[1]: {
        'frequency_range': [0.14, 0.45],
        'energy_range': [10.3, 10.9],
        'quadratic_scaling': 0.08,
        'linear_scaling': 0.04,
        'nonadiabatic': False,
        'numStates': 4,
        'numModes': 6,
        },

    # 7x12
    systems.name_lst[2]: {
        'frequency_range': [0.1, 0.39],
        'energy_range': [14.0, 14.8],
        'quadratic_scaling': 0.08,
        'linear_scaling': 0.04,
        'nonadiabatic': False,
        'numStates': 7,
        'numModes': 12,
        }
}


def generate_input_file(name):
    """ to modify the initial *.json or *.op input files """
    src = join(abspath(dirname(__file__)), "input_json")
    path = join(src, f"model_{name:s}.json")
    d = vIO.generate_vibronic_model_data(parameters[name])
    vIO.save_model_to_JSON(path, d)
    return


if (__name__ == "__main__"):
    # the 2x2 model is hand built - not generated
    for name in systems.name_lst[1:]:
        generate_input_file(name)
