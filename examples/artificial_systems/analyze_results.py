# analyze_results.py - creates all the directories and copies in the necessary files

# system imports
import os

# third party imports

# local imports
import context
import systems
from pibronic.stats import stats
import pibronic.data.file_structure as fs


def simple_wrapper(FS):
    # TODO - sort out which ones I want to use
    stats.statistical_analysis_of_pimc(FS.path_root, FS.id_data, FS.id_rho, method="basic")
    # stats.statistical_analysis_of_pimc(FS.path_root, FS.id_data, FS.id_rho, method="alpha")
    stats.jackknife_analysis_of_pimc(FS.path_root, FS.id_data, FS.id_rho, method="basic")
    # stats.jackknife_analysis_of_pimc(FS.path_root, FS.id_data, FS.id_rho, method="alpha")
    return


def automate_wrapper(name):
    """ loops over the data sets and different rhos """
    systems.assert_system_name_is_valid(name)

    root = context.choose_root_folder()

    for id_data in systems.id_dict[name]:
        for id_rho in systems.rho_dict[name][id_data]:
            # TODO - this is a perfect example where FS should have a method which returns a bool
            # which tells if all the directories or necessary - WITHOUT CREATING THE DIRECTORIES
            # -- this alternative solution of passing in a bool flag might be a reasonable approach
            FS = fs.FileStructure(root, id_data, id_rho, no_makedir=True)
            if os.path.isfile(FS.path_rho_model):
                simple_wrapper(FS)
    return


if (__name__ == "__main__"):
    # eventual code
    # map(automate_wrapper, systems.name_lst)

    # during testing
    # Sequential, comment out lines if you only need to run for individual models
    automate_wrapper(systems.name_lst[0])
    automate_wrapper(systems.name_lst[1])
    automate_wrapper(systems.name_lst[2])
