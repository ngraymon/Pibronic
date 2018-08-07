# generate_energy_plots.py

# system imports
from multiprocessing import Process

# third party imports

# local imports
import context
import systems
from pibronic.plotting import grid_plots as gp
from pibronic.data import file_structure as fs


def automate_energy_grid_plots(name):
    """ loops over the data sets generating energy grid plots for each one  """
    systems.assert_system_name_is_valid(name)

    root = context.choose_root_folder()

    for id_data in systems.id_dict[name]:
        FS = fs.FileStructure(root, id_data)
        plotObj = gp.plotGrids(FS)
        plotObj.load_data()
        plotObj.plot_energy()
        plotObj.plot_linear_coupling()
        print(f"Finished plotting D{id_data:d}")
    return


if (__name__ == "__main__"):

    # If you want to speed things up you can split the work across four processes
    # pool is not used because these processes are I/O intensive and we want them to run concurrently
    multiprocessing_flag = True

    if multiprocessing_flag:
        # create a thread for each system
        lst_p = [Process(target=automate_energy_grid_plots,
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
        automate_energy_grid_plots(systems.name_lst[0])
        automate_energy_grid_plots(systems.name_lst[1])
        automate_energy_grid_plots(systems.name_lst[2])
        pass

    print("Done plotting")
