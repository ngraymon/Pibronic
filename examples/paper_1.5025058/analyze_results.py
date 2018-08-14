# analyze_results.py - creates all the directories and copies in the necessary files

# system imports
import os
from multiprocessing import Process

# third party imports

# local imports
import context
import systems
from pibronic.stats import stats
import pibronic.data.file_structure as fs


def simple_wrapper(FS):
    # TODO - sort out which ones I want to use
    stats.statistical_analysis_of_pimc(FS, method="basic")
    # stats.statistical_analysis_of_pimc(FS, method="alpha")
    stats.jackknife_analysis_of_pimc(FS, method="basic")
    # stats.jackknife_analysis_of_pimc(FS, method="alpha")
    return


def automate_statistical_analysis(name):
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

    # If you want to speed things up you can split the work across four processes
    # pool is not used because these processes can be I/O intensive and we want them to run concurrently
    multiprocessing_flag = True

    if multiprocessing_flag:
        # create a thread for each system
        lst_p = [Process(target=automate_statistical_analysis,
                         args=(name,)
                         ) for name in systems.name_lst
                 ]
        # start the threads
        for p in lst_p:
            p.start()

        # wait until they are all finished
        for p in lst_p:
            p.join()

    else:
        # Sequential, comment out lines if you only need to run for individual models
        automate_statistical_analysis("superimposed")
        automate_statistical_analysis("displaced")
        automate_statistical_analysis("elevated")
        automate_statistical_analysis("jahnteller")
