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
    multiprocessing_flag = False

    if multiprocessing_flag:
        lst_p = [0]*len(systems.name_lst)

        for idx in range(len(systems.name_lst)):
            lst_p[idx] = Process(target=automate_energy_grid_plots, args=(systems.name_lst[idx],))

        for p in lst_p:
            p.start()

        for p in lst_p:
            p.join()
    else:
        automate_energy_grid_plots(systems.name_lst[0])
        automate_energy_grid_plots(systems.name_lst[1])
        automate_energy_grid_plots(systems.name_lst[2])
    print("Done plotting")
