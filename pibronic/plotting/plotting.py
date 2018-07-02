""" first pass at creating plotting module """


# system imports
import json
from os.path import join

# third party imports
import numpy as np
from numpy import float64 as F64
import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.cm as cmap

# local imports
from ..data import postprocessing as pp
from ..data import file_structure as fs
from ..constants import beta
# from ..log_conf import log
# from ..vibronic import vIO, VMK


def prepare_mpl_rc_file(pretty_but_slow=False):
    """ TODO - this needs to be refactored and cleaned up (it is sufficiently functional for the moment) """

    # TODO - this doesn't seem to work?
    # mpl.rcParams['backend'] = "agg"  # For the server we need to force the use of Agg
    # mpl.rcParams['backend'] = "ps"  # We need to use the postscript backend to generate eps files

    plt.switch_backend("agg")

    if pretty_but_slow:
        # change the font
        mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
        # mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Sans serif']})
        # mpl.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    mpl.rc('text', usetex=True)  # using LaTeX
    # we need to load the amsmath package to use the \text{} command
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    return


def load_latex_module_on_server():
    """ load the texlive module so that we can make plots with latex
    this function will only work on our local server
    TODO - there should be a replacement for local execution and execution on other servers"""
    import subprocess  # replace with a from subprocess import?
    cmd = ['/usr/bin/modulecmd', 'python', 'load', 'texlive/2017']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, error = p.communicate()
    exec(out)  # this is necessary!
    return


class plotVirtual:
    """ outline the basic flow of plotting
    most members are designed to be overloaded
    actual plotting functions should be added to children classes
    """

    def generate_file_lists(self):
        """ create lists of paths to all data files to be used for plotting """
        return

    def generate_parameter_lists(self):
        """ create lists of all possible unique valid parameters that are to be plotted
        for example:
            a list of all possible bead values might be [12, 20, 50],
            a list of all possible temperature values might be [250.00, 275.00, 300.00]
        which could arise from 3 data files with the following parameters:
            [12, 250.00], [20, 275.00], [50, 300.00]
        or 5 data files with the following parameters:
            [12, 250.00], [12, 275.00], [12, 300.00], [22, 300.00], [50, 300.00]

        the purposes of this function is to generate lists which allow for modifying the range of each parameter separately using intersections

        """
        return

    def validate_data(self):
        return

    def __init__(self, list_of_FileStructure_objects):
        """ x """

        # if we are provided with just one FileStructure object then wrap it in a list
        if isinstance(list_of_FileStructure_objects, fs.FileStructure):
            list_of_FileStructure_objects = [list_of_FileStructure_objects, ]

        self.FS_lst = list_of_FileStructure_objects

        for FS in self.FS_lst:
            FS.generate_model_hashes()  # build the hashes so that we can check against them

        self.generate_file_lists()
        self.generate_parameter_lists()
        # might want to call these functions seperately from initialization
        # self.validate_data()  # this one will be tricky - might be optional
        # prepare_mpl_rc_file()
        return


class plot_Z_multiple_FS(plotVirtual):
    """ this is just an empty template at the moment, will be completed later"""

    def generate_file_lists(self):
        """ x """

        # we create a list of lists of paramters specific to each FileStructure object
        self.list_jackknife = [[] for _ in range(len(self.FS_lst))]
        for idx, FS in enumerate(self.FS_lst):
            self.list_jackknife[idx] = pp.retrive_jackknife_file_list(FS)
            self.list_jackknife[idx] = pp.prune_results_using_hashes(FS, self.list_jackknife[idx])
        return

    def generate_parameter_lists(self):
        """ x """

        # lots of issues can arise from this lists of lists approach
        # especially if the order is implicit and the code relies on that!

        func = pp.extract_bead_value_from_thermo_file_path
        self.lst_P = [list(set(map(func, lst))) for lst in self.list_jackknife]

        func = pp.extract_temperature_value_from_thermo_file_path
        self.lst_T = [list(set(map(func, lst))) for lst in self.list_jackknife]

        func = pp.extract_sample_value_from_thermo_file_path
        self.lst_X = [list(set(map(func, lst))) for lst in self.list_jackknife]

        # sort them -- TODO - this won't work?
        map(list.sort, self.lst_P)
        map(list.sort, self.lst_T)
        map(list.sort, self.lst_X)

        return

    def __init__(self, list_of_FileStructure_objects):
        """ x """

        super().__init__(list_of_FileStructure_objects)

    def load_data(self):
        """ x """

        # create the array where we store the data
        dims = (len(self.lst_P), len(self.lst_T), len(self.lst_X))

        dicttype = np.dtype({
            'names': ["Z", "Z error", "E", "E error", "Cv", "Cv error",
                      "jk_E", "jk_E error", "jk_Cv", "jk_Cv error"],
            'formats': [F64, F64, F64, F64, F64, F64, F64, F64, F64, F64],
            })

        arr = np.full(dims, np.nan, dtype=dicttype)

        for path in self.list_jackknife:
            print(path)
            P = pp.extract_bead_value_from_thermo_file_path(path)
            T = pp.extract_temperature_value_from_thermo_file_path(path)
            X = pp.extract_sample_value_from_thermo_file_path(path)

            idx_P = self.lst_P.index(P)
            idx_T = self.lst_T.index(T)
            idx_X = self.lst_X.index(X)

            # bit of a naieve way to load the data?
            # should possibly do some validation?
            with open(path, 'r') as file:
                data = json.loads(file.read())

            for key in data:
                if key != "hash_vib" and key != "hash_rho":
                    arr[idx_P, idx_T, idx_X][key] = data[key]
        return


class plot_Z_test(plotVirtual):
    """ plotting when we only provide 1 FS object"""

    def generate_file_lists(self):
        """ create a list of lists of paramters specific to each FileStructure object """
        self.list_jackknife = [[]]
        for FS in self.FS_lst:
            self.list_jackknife[0] = pp.retrive_jackknife_file_list(FS)
            self.list_jackknife[0] = pp.prune_results_using_hashes(FS, self.list_jackknife[0])
        return

    def generate_parameter_lists(self):
        """ x """

        # lots of issues can arise from this lists of lists approach
        # especially if the order is implicit and the code relies on that!

        func = pp.extract_bead_value_from_thermo_file_path
        self.lst_P = [list(set(map(func, self.list_jackknife[0])))]

        func = pp.extract_temperature_value_from_thermo_file_path
        self.lst_T = [list(set(map(func, self.list_jackknife[0])))]

        func = pp.extract_sample_value_from_thermo_file_path
        self.lst_X = [list(set(map(func, self.list_jackknife[0])))]

        # sort them
        for lst in self.lst_P:
            lst.sort()
        for lst in self.lst_T:
            lst.sort()
        for lst in self.lst_X:
            lst.sort()

        return

    def __init__(self, list_of_FileStructure_objects):
        """ x """
        assert isinstance(list_of_FileStructure_objects, fs.FileStructure), " this subclass only takes 1 FS"
        super().__init__(list_of_FileStructure_objects)

    def load_data(self):
        """ x """

        # create the array where we store the data
        dims = (len(self.lst_P[0]), len(self.lst_T[0]), len(self.lst_X[0]))

        dicttype = np.dtype({
            'names': ["Z", "Z error", "E", "E error", "Cv", "Cv error",
                      "jk_E", "jk_E error", "jk_Cv", "jk_Cv error"],
            'formats': [F64, F64, F64, F64, F64, F64, F64, F64, F64, F64],
            })

        self.arr = np.full(dims, np.nan, dtype=dicttype)

        for path in self.list_jackknife[0]:
            P = pp.extract_bead_value_from_thermo_file_path(path)
            T = pp.extract_temperature_value_from_thermo_file_path(path)
            X = pp.extract_sample_value_from_thermo_file_path(path)

            idx_P = self.lst_P[0].index(P)
            idx_T = self.lst_T[0].index(T)
            idx_X = self.lst_X[0].index(X)

            # bit of a naieve way to load the data?
            # should possibly do some validation?
            with open(path, 'r') as file:
                data = json.loads(file.read())

            for key in data:
                if key != "hash_vib" and key != "hash_rho":
                    self.arr[idx_P, idx_T, idx_X][key] = data[key]
        return

    # TODO - this could be improved
    def generate_tau_values(self, temperature):
        """ returns a numpy array of the same length as lst_P
        takes in one temperature and an array of P values"""
        tau_arr = np.full(len(self.lst_P[0]), fill_value=beta(temperature))
        tau_arr /= self.lst_P[0]
        return tau_arr

    def plot_Z(self):
        fig, ax = plt.subplots(1, 1)

        # HACKY - this won't work anymore with the nested lists
        if len(self.lst_T[0]) is 1:
            ax = [ax, ]

        # print Z values
        for T in self.lst_T[0]:
            idx_T = self.lst_T[0].index(T)
            tau_values = self.generate_tau_values(T)
            for X in self.lst_X[0]:
                idx_X = self.lst_X[0].index(X)

                x = tau_values.view()
                y = self.arr[:, idx_T, idx_X]["Z"].view()
                yerr = self.arr[:, idx_T, idx_X]["Z error"].view()
                label = "Z (T={:.2f}K) (X={:.1E})".format(T, X)

                ax[idx_T].errorbar(x, y, xerr=None, yerr=yerr, marker='o', label=label)
                ax[idx_T].legend()

        fig.suptitle("Z MC")
        fig.set_size_inches(10, 10)
        filename = "Z_D{:d}_R{:d}.pdf".format(self.FS_lst[0].id_data, self.FS_lst[0].id_rho)
        path_out = join(self.FS_lst[0].path_rho_plots, filename)
        fig.savefig(path_out, transparent=True, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        return

    def plot_E(self):
        fig, ax = plt.subplots(1, 1)

        # HACKY - this won't work anymore with the nested lists
        if len(self.lst_T[0]) is 1:
            ax = [ax, ]

        # print E values
        for T in self.lst_T[0]:
            idx_T = self.lst_T[0].index(T)
            tau_values = self.generate_tau_values(T)
            for X in self.lst_X[0]:
                idx_X = self.lst_X[0].index(X)

                x = tau_values.view()
                y = self.arr[:, idx_T, idx_X]["E"].view()
                yerr = self.arr[:, idx_T, idx_X]["E error"].view()
                label = "E (T={:.2f}K) (X={:.1E})".format(T, X)

                ax[idx_T].errorbar(x, y, xerr=None, yerr=yerr, marker='o', label=label)
                ax[idx_T].legend()

        fig.suptitle("Energy")
        fig.set_size_inches(10, 10)
        filename = "E_D{:d}_R{:d}.pdf".format(self.FS_lst[0].id_data, self.FS_lst[0].id_rho)
        path_out = join(self.FS_lst[0].path_rho_plots, filename)
        fig.savefig(path_out, transparent=True, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        return

    def plot_Cv(self):
        fig, ax = plt.subplots(1, 1)

        # HACKY - this won't work anymore with the nested lists
        if len(self.lst_T[0]) is 1:
            ax = [ax, ]

        # print Cv values
        for T in self.lst_T[0]:
            idx_T = self.lst_T[0].index(T)
            tau_values = self.generate_tau_values(T)
            for X in self.lst_X[0]:
                idx_X = self.lst_X[0].index(X)

                x = tau_values.view()
                y = self.arr[:, idx_T, idx_X]["Cv"].view()
                yerr = self.arr[:, idx_T, idx_X]["Cv error"].view()
                label = "Cv (T={:.2f}K) (X={:.1E})".format(T, X)

                ax[idx_T].errorbar(x, y, xerr=None, yerr=yerr, marker='o', label=label)
                ax[idx_T].legend()

        fig.suptitle("Cv")
        fig.set_size_inches(10, 10)
        filename = "Cv_D{:d}_R{:d}.pdf".format(self.FS_lst[0].id_data, self.FS_lst[0].id_rho)
        path_out = join(self.FS_lst[0].path_rho_plots, filename)
        fig.savefig(path_out, transparent=True, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        return

    def plot(self):
        """ x """

        # necessary for now
        prepare_mpl_rc_file()
        load_latex_module_on_server()

        self.plot_Z()
        self.plot_E()
        self.plot_Cv()
        return


if (__name__ == "__main__"):
    pass
