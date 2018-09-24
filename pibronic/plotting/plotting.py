""" first pass at creating plotting module """


# system imports
import json
from os.path import join

# third party imports
import numpy as np
from numpy import float64 as F64
import matplotlib.pyplot as plt

# local imports
from .server import prepare_mpl_rc_file, load_latex_module_on_server
from .virtual import plotVirtual
from ..data import postprocessing as pp
from ..data import file_structure as fs
from ..constants import beta
from ..vibronic import vIO



def load_analytical_original(self):
    """ generic loading function that can be used by multiple plotting classes """

    # create the list of dictionaries to hold the original analytical data
    self.analytical_orig_list = [{} for _ in range(len(self.FS_lst))]

    # load that data into each dictionary in the list
    for idx_FS, FS in enumerate(self.FS_lst):

        with open(FS.path_analytic_orig, 'r') as file:
            data = json.loads(file.read())

        for T in self.lst_T[idx_FS]:
            self.analytical_orig_list[idx_FS][f"{T:.2f}"] = data[f"{T:.2f}"]

    return


def load_analytical_sampling(self):
    """ generic loading function that can be used by multiple plotting classes"""

    # create the list of dictionaries to hold the sampling analytical data
    self.analytical_rho_list = [{} for _ in range(len(self.FS_lst))]

    # load that data into each dictionary in the list
    for idx_FS, FS in enumerate(self.FS_lst):

        with open(FS.path_analytic_rho, 'r') as file:
            data = json.loads(file.read())

        for T in self.lst_T[idx_FS]:
            self.analytical_rho_list[idx_FS][f"{T:.2f}"] = data[f"{T:.2f}"]

    return


def load_sos(self, ):
    """ generic loading function that can be used by multiple plotting classes"""

    # create the list of dictionaries to hold the sum-over-states data
    self.sos_list = [{} for _ in range(len(self.FS_lst))]

    # temporary constants to avoid any errors
    # this will be changed in the future
    MAX_BASIS_SIZE = 80

    # load that data into each dictionary in the list
    for idx_FS, FS in enumerate(self.FS_lst):
        path = FS.template_sos_vib.format(B=MAX_BASIS_SIZE)

        with open(path, 'r') as file:
            data = json.loads(file.read())

        for T in self.lst_T[idx_FS]:
            self.sos_list[idx_FS][f"{T:.2f}"] = data[f"{T:.2f}"]

    return


def load_trotter_coupled(self):
    """ generic loading function that can be used by multiple plotting classes"""

    # create the list of dictionaries to hold the coupled trotter data
    self.trotter_coupled_list = [{} for _ in range(len(self.FS_lst))]

    # temporary constants to avoid any errors
    # this will be changed in the future
    MAX_BASIS_SIZE = 80

    # load that data into each dictionary in the list
    for idx_FS, FS in enumerate(self.FS_lst):
        for idx_P, P, in enumerate(self.lst_P[idx_FS]):
            path = FS.template_trotter_vib.format(P=P, B=MAX_BASIS_SIZE)

            with open(path, 'r') as file:
                data = json.loads(file.read())

            temp_dict = {}
            for T in self.lst_T[idx_FS]:
                temp_dict[f"{T:.2f}"] = data[f"{T:.2f}"]

            # TODO - is there possibly a better way of doing this?
            self.trotter_coupled_list[idx_FS].update({P: temp_dict})

    return



# TODO - this function could be improved
def generate_tau_values(self, temperature, idx_FS):
    """ returns a numpy array of the same length as lst_P
    takes in one temperature and an array of P values"""
    tau_arr = np.full(len(self.lst_P[idx_FS]), fill_value=beta(temperature))
    tau_arr /= self.lst_P[idx_FS]
    return tau_arr


class plot_Z_multiple_FS(plotVirtual):
    """ this is just an empty template at the moment, will be completed later"""

    def generate_file_lists(self):
        """ x """

        # we create a list of lists of parameters specific to each FileStructure object
        self.list_jackknife = [[] for _ in range(len(self.FS_lst))]
        for idx_FS, FS in enumerate(self.FS_lst):
            self.list_jackknife[idx_FS] = pp.retrive_jackknife_file_list(FS)
            self.list_jackknife[idx_FS] = pp.prune_results_using_hashes(FS, self.list_jackknife[idx_FS])

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

        # sort them -- TODO - it would be nice if we could use map() here
        for lst in self.lst_P:
            lst.sort()
        for lst in self.lst_T:
            lst.sort()
        for lst in self.lst_X:
            lst.sort()

    def __init__(self, list_of_FileStructure_objects):
        """ x """
        super().__init__(list_of_FileStructure_objects)

    def load_pimc_data(self):
        """ x """

        # a list of numpy arrays, each one corresponds to a FS in the self.FS_lst
        self.arr = [[] for _ in range(len(self.FS_lst))]

        # we will store the specific dims of each array associated with a FS
        dims = [[] for _ in range(len(self.FS_lst))]

        # the data structure of a single entry of the numpy arrays in the list self.arr
        dicttype = np.dtype({
            'names': ["Z", "Z error", "E", "E error", "Cv", "Cv error",
                      "jk_E", "jk_E error", "jk_Cv", "jk_Cv error"],
            'formats': [F64, F64, F64, F64, F64, F64, F64, F64, F64, F64],
            })

        # create the array where we store the data
        for idx_FS, FS in enumerate(self.FS_lst):

            dims[idx_FS] = (len(self.lst_P[idx_FS]),
                            len(self.lst_T[idx_FS]),
                            len(self.lst_X[idx_FS])
                            )

            self.arr[idx_FS] = np.full(dims[idx_FS], np.nan, dtype=dicttype)

        for idx_FS, FS in enumerate(self.FS_lst):
            for path in self.list_jackknife[idx_FS]:
                P = pp.extract_bead_value_from_thermo_file_path(path)
                T = pp.extract_temperature_value_from_thermo_file_path(path)
                X = pp.extract_sample_value_from_thermo_file_path(path)

                if P not in self.lst_P[idx_FS]:
                    continue
                idx_P = self.lst_P[idx_FS].index(P)
                idx_T = self.lst_T[idx_FS].index(T)
                idx_X = self.lst_X[idx_FS].index(X)

                # bit of a naive way to load the data?
                # should possibly do some validation?
                with open(path, 'r') as file:
                    data = json.loads(file.read())

                for key in data:
                    if key != "hash_vib" and key != "hash_rho":
                        self.arr[idx_FS][idx_P, idx_T, idx_X][key] = data[key]
        return

    def load_data(self):
        """ x """
        self.load_pimc_data()
        return


class plot_original_Z_test(plotVirtual):
    """ plotting pimc results against analytical results of original model
    when we only provide 1 FS object"""

    def generate_file_lists(self):
        """ create a list of lists of parameters specific to each FileStructure object """
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

    def load_pimc_data(self):
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

            # bit of a naive way to load the data?
            # should possibly do some validation?
            with open(path, 'r') as file:
                data = json.loads(file.read())

            for key in data:
                if key != "hash_vib" and key != "hash_rho":
                    self.arr[idx_P, idx_T, idx_X][key] = data[key]

    def load_analytical_data(self):
        """ x """

        # assuming we only have one FS in the list
        FS = self.FS_lst[0]

        self.analytical_orig_dict = {}
        self.analytical_rho_dict = {}

        with open(FS.path_analytic_orig, 'r') as file:
            data = json.loads(file.read())
            for T in self.lst_T[0]:
                self.analytical_orig_dict[f"{T:.2f}"] = data[f"{T:.2f}"]

        with open(FS.path_analytic_rho, 'r') as file:
            data = json.loads(file.read())
            for T in self.lst_T[0]:
                self.analytical_rho_dict[f"{T:.2f}"] = data[f"{T:.2f}"]

        return

    def prepare_data(self):
        # we need to modify Z using the analytical rho (Z_rho)
        for T in self.lst_T[0]:
            Z_rho = self.analytical_rho_dict[f"{T:.2f}"]["Z_sampling"]
            idx_T = self.lst_T[0].index(T)
            self.arr[:, idx_T, :]["Z"] *= Z_rho
            self.arr[:, idx_T, :]["Z error"] *= Z_rho

        # we might also want to modify it to print the percent error
        self.percent_error = True
        if self.percent_error:
            for T in self.lst_T[0]:
                idx_T = self.lst_T[0].index(T)
                self.arr[:, idx_T, :]["Z"] -= self.analytical_orig_dict[f"{T:.2f}"]["Z_coupled"]
                self.arr[:, idx_T, :]["Z"] *= 100.0
                self.arr[:, idx_T, :]["Z"] /= self.analytical_orig_dict[f"{T:.2f}"]["Z_coupled"]
                self.arr[:, idx_T, :]["Z error"] *= 100.0
                self.arr[:, idx_T, :]["Z error"] /= self.analytical_orig_dict[f"{T:.2f}"]["Z_coupled"]
        return

    def load_data(self):
        """ x """
        self.load_pimc_data()
        self.load_analytical_data()
        self.prepare_data()
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

            true_answer = self.analytical_orig_dict[f"{T:.2f}"]["Z_coupled"]
            if self.percent_error:
                true_answer = 0.0

            ax[idx_T].axhline(y=true_answer, linewidth=2, color='r',
                              label='Analytically derived Z from original model'
                              )

        #
        x_label = r'$\tau\,(\text{eV}^{-1})$'
        ax[0].set_xlabel(x_label)
        ax[0].set_xscale('log')

        # plt.minorticks_off() # turns off minor ticks that are added with a log plot
        # points BOTH ticks inward
        ax[0].tick_params(which='both', direction='in', pad=2)

        # Remove the plot frame lines. They are unnecessary chart junk.
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)

        y_label = r"$\frac{Z_g}{Z_\varrho}$"
        if self.percent_error:
            y_label = r"\% Difference    $\frac{Z_g}{Z_\varrho}$"

        ax[0].set_ylabel(y_label)

        # plot title
        A, N = vIO.extract_dimensions_of_diagonal_model(self.FS_lst[0])
        plot_title = r'$\tau$ convergence'
        plot_title += f"\nData Set {self.FS_lst[0].id_data:d}"
        plot_title += f"\n{A:d} surfaces {N:d} normal modes"
        fig.suptitle(plot_title)

        fig.set_size_inches(10, 10)
        filename = "Z_orig_D{:d}_R{:d}.pdf".format(self.FS_lst[0].id_data, self.FS_lst[0].id_rho)
        path_out = join(self.FS_lst[0].path_rho_plots, filename)
        fig.savefig(path_out, transparent=True, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        return

    def plot(self):
        """ x """

        # necessary for now
        prepare_mpl_rc_file(pretty_but_slow=True)
        load_latex_module_on_server()

        self.plot_Z()

        return


# should we rename this?
class plot_original_Z_vs_diagonal_test(plot_Z_multiple_FS):
    """ plotting pimc results drawn from the original_coupled_model.json vs the pimc results drawn from the diagonal of the coupled_model.json vs analytical results of the original model
    when we provide 2 FS objects"""

    def __init__(self, list_of_FileStructure_objects):
        """ x """
        assert len(list_of_FileStructure_objects) is 2, " this subclass takes exactly 2 FS"
        super().__init__(list_of_FileStructure_objects)

    def load_analytical_data(self):
        """ x """
        load_analytical_original(self)
        load_analytical_sampling(self)

    def prepare_data(self):
        # we need to modify Z using the analytical rho (Z_rho)

        for idx_FS, FS in enumerate(self.FS_lst):
            for T in self.lst_T[idx_FS]:
                Z_rho = self.analytical_rho_list[idx_FS][f"{T:.2f}"]["Z_sampling"]
                idx_T = self.lst_T[idx_FS].index(T)
                self.arr[idx_FS][:, idx_T, :]["Z"] *= Z_rho
                self.arr[idx_FS][:, idx_T, :]["Z error"] *= Z_rho

        # we might also want to modify it to print the percent error
        self.percent_error = True

        if not self.percent_error:
            return

        for idx_FS, FS in enumerate(self.FS_lst):
            for X in self.lst_X[idx_FS]:  # don't plot more than the lowest # of samples
                idx_X = self.lst_X[idx_FS].index(X)
                for T in self.lst_T[idx_FS]:
                    idx_T = self.lst_T[idx_FS].index(T)

                    view = self.arr[idx_FS][:, idx_T, idx_X].view()
                    lytical = self.analytical_orig_list[idx_FS][f"{T:.2f}"]["Z_coupled"]

                    view["Z"] -= lytical
                    view["Z"] *= 100.0
                    view["Z"] /= lytical

                    view["Z error"] *= 100.0
                    view["Z error"] /= lytical
        return

    def load_data(self):
        """ x """
        self.load_pimc_data()
        self.load_analytical_data()
        self.prepare_data()
        return

    # TODO - this could be improved
    def generate_tau_values(self, temperature, idx_FS):
        """ returns a numpy array of the same length as lst_P
        takes in one temperature and an array of P values"""
        tau_arr = np.full(len(self.lst_P[idx_FS]), fill_value=beta(temperature))
        tau_arr /= self.lst_P[idx_FS]
        return tau_arr

    def plot_Z(self):
        fig, ax = plt.subplots(1, 1)

        if not isinstance(ax, list):
            ax = [ax, ]

        labels = {0: "Z (id\_rho={:d}) (X={:.1E}) diagonal of transformed matrix ",
                  1: "Z (id\_rho={:d}) (X={:.1E}) original coupled model ",
                  2: "Z (id\_rho={:d}) (X={:.1E}) iterative model ",
                  3: "Z (id\_rho={:d}) (X={:.1E}) wrong re-weighted iterative model",
                  4: "Z (id\_rho={:d}) (X={:.1E}) correct re-weighted iterative model",
                  }

        # print Z values
        for idx_FS, FS in enumerate(self.FS_lst):
            for T in self.lst_T[idx_FS]:
                idx_T = self.lst_T[idx_FS].index(T)
                tau_values = self.generate_tau_values(T, idx_FS)
                # for X in self.lst_X[idx_FS]:  # don't plot more than the lowest # of samples
                X = self.lst_X[idx_FS][-1]
                idx_X = self.lst_X[idx_FS].index(X)

                x = tau_values.view()
                y = self.arr[idx_FS][:, idx_T, idx_X]["Z"].view()
                yerr = self.arr[idx_FS][:, idx_T, idx_X]["Z error"].view()
                label = labels[FS.id_rho].format(FS.id_rho, X)

                ax[idx_T].errorbar(x, y, xerr=None, yerr=yerr,
                                   marker='o' if idx_FS == 0 else 'x',
                                   markerfacecolor="None",
                                   label=label)
                ax[idx_T].legend()

                true_answer = self.analytical_orig_list[idx_FS][f"{T:.2f}"]["Z_coupled"]
                if self.percent_error:
                    true_answer = 0.0

                ax[idx_T].axhline(y=true_answer, linewidth=2, color='r',
                                  label='Analytically derived Z from original model'
                                  )

        # Add an inset to the plot!
        if self.FS_lst[0].id_data >= 10:
            from mpl_toolkits.axes_grid1.inset_locator import mark_inset
            from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

            # Build and manually place the inset (for the rho_1 case )

            # ax[0].xaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(11, 10)))

            ax_ins = fig.add_axes([0, 0, 1, 1], label='inset 1')
            ip = InsetPosition(ax[0], [0.1, 0.3, 0.5, 0.6])  # [%left, %up, %width, %height]
            ax_ins.set_axes_locator(ip)
            # Connect the inset to its respective region
            mark_inset(ax[0], ax_ins, loc1=3, loc2=4, fc="none", ec="0.5")

            # window dressing?
            # ax_ins.set_xscale('log')
            ax_ins.get_xaxis().set_visible(False)
            ax_ins.tick_params(which='both', direction='in', pad=4)
            # ax_ins.spines['top'].set_visible(False)

            # HACK CONSTANTS
            # F_index = -1
            idx_T = 0
            idx_X = -1
            T = 300.00
            # dataView = self.arr[F_index, R_index, T_index, 0, :].view()
            slice_insert = np.s_[-1:-2:-1]
            # if np.any(np.isnan(dataView)):
            #     first_Finite = np.where(np.isfinite(self.arr[F_index, R_index, T_index, 0, :]))[0][-1]
            #     slice_insert = np.s_[first_Finite:first_Finite-10:-1]
            # --------------------------------------------------------------------------------
            # Plot the first FS
            idx_FS = 0
            tau_values = self.generate_tau_values(T, idx_FS)
            x = tau_values[slice_insert].view()
            y = self.arr[idx_FS][slice_insert, idx_T, idx_X]["Z"].view()
            yerr = self.arr[idx_FS][slice_insert, idx_T, idx_X]["Z error"].view()
            label = labels[idx_FS].format(self.FS_lst[idx_FS].id_rho, self.lst_X[idx_FS][0])
            ax_ins.errorbar(x, y, xerr=None, yerr=yerr,
                            marker='o' if idx_FS == 0 else 'x',
                            markerfacecolor="None",
                            label=label)
            # --------------------------------------------------------------------------------
            # Plot the second FS
            idx_FS = 1
            tau_values = self.generate_tau_values(T, idx_FS)
            x = tau_values[slice_insert].view()
            y = self.arr[idx_FS][slice_insert, idx_T, idx_X]["Z"].view()
            yerr = self.arr[idx_FS][slice_insert, idx_T, idx_X]["Z error"].view()
            label = labels[idx_FS].format(self.FS_lst[idx_FS].id_rho, self.lst_X[idx_FS][0])
            ax_ins.errorbar(x, y, xerr=None, yerr=yerr,
                            marker='o' if idx_FS == 0 else 'x',
                            markerfacecolor="None",
                            label=label)
            # --------------------------------------------------------------------------------
            # REFERENCE / EXACT ANSWER
            true_answer = self.analytical_orig_list[0][f"{T:.2f}"]["Z_coupled"]
            if self.percent_error:
                true_answer = 0.0

            ax_ins.axhline(y=true_answer, linewidth=2, color='r',
                           label='Analytically derived Z from original model'
                           )
            # --------------------------------------------------------------------------------

        #
        x_label = r'$\tau\,(\text{eV}^{-1})$'
        ax[0].set_xlabel(x_label)
        ax[0].set_xscale('log')

        # plt.minorticks_off() # turns off minor ticks that are added with a log plot
        # points BOTH ticks inward
        ax[0].tick_params(which='both', direction='in', pad=2)

        # Remove the plot frame lines. They are unnecessary chart junk.
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)

        # y_label = r"$\frac{Z_g}{Z_\varrho}$"
        y_label = r"$Z_H$"
        if self.percent_error:
            # y_label = r"\% Difference    $\frac{Z_g}{Z_\varrho}$"
            y_label = r"\% Difference    $Z_H$"

        ax[0].set_ylabel(y_label)
        # ax[0].set_yscale('log')

        # plot title
        A, N = vIO.extract_dimensions_of_diagonal_model(self.FS_lst[0])
        plot_title = r'$\tau$ convergence'
        plot_title += f"\nData Set {self.FS_lst[0].id_data:d}"
        plot_title += f"\n{A:d} surfaces {N:d} normal modes"
        fig.suptitle(plot_title)

        fig.set_size_inches(10, 10)
        filename = "Z_orig_D{:d}_R{:d}_R{:d}.pdf".format(self.FS_lst[0].id_data, self.FS_lst[0].id_rho, self.FS_lst[1].id_rho)
        path_out = join(self.FS_lst[0].path_vib_plots, filename)
        fig.savefig(path_out, transparent=True, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        return

    def plot(self):
        """ x """

        # necessary for now
        prepare_mpl_rc_file(pretty_but_slow=True)
        load_latex_module_on_server()

        self.plot_Z()

        return


class plot_sos_Z_vs_rho_n(plot_Z_multiple_FS):
    """ plotting pimc results drawn from the coupled_model.json vs the pimc results drawn from a rho vs sos results of the coupled model when we provide 2 FS objects"""

    def __init__(self, list_of_FileStructure_objects):
        """ x """
        assert len(list_of_FileStructure_objects) is 2, " this subclass takes exactly 2 FS"
        super().__init__(list_of_FileStructure_objects)

    def prepare_data(self):
        # we need to modify Z using the analytical rho (Z_rho)

        for idx_FS, FS in enumerate(self.FS_lst):
            for T in self.lst_T[idx_FS]:
                Z_rho = self.analytical_rho_list[idx_FS][f"{T:.2f}"]["Z_sampling"]
                idx_T = self.lst_T[idx_FS].index(T)
                self.arr[idx_FS][:, idx_T, :]["Z"] *= Z_rho
                self.arr[idx_FS][:, idx_T, :]["Z error"] *= Z_rho

        # we might also want to modify it to print the percent error
        self.percent_error = True

        if not self.percent_error:
            return

        for idx_FS, FS in enumerate(self.FS_lst):
            for X in self.lst_X[idx_FS]:  # don't plot more than the lowest # of samples
                idx_X = self.lst_X[idx_FS].index(X)
                for T in self.lst_T[idx_FS]:
                    idx_T = self.lst_T[idx_FS].index(T)

                    view = self.arr[idx_FS][:, idx_T, idx_X].view()
                    sos = self.sos_params[f"{T:.2f}"]["Z_coupled"]

                    view["Z"] -= sos
                    view["Z"] *= 100.0
                    view["Z"] /= sos

                    view["Z error"] *= 100.0
                    view["Z error"] /= sos
        return

    def load_data(self):
        """ x """
        self.load_pimc_data()
        load_sos(self)
        load_analytical_sampling(self)
        self.prepare_data()

    def plot_Z(self):
        fig, ax = plt.subplots(1, 1)

        if not isinstance(ax, list):
            ax = [ax, ]

        labels = {0: "Z (id\_rho={:d}) (X={:.1E}) diagonal of transformed matrix ",
                  1: "Z (id\_rho={:d}) (X={:.1E}) alternative sampling distribution from paper",
                  2: "Z (id\_rho={:d}) (X={:.1E}) 2nd alternative sampling distribution",
                  10: "Z (id\_rho={:d}) (X={:.1E}) iterative model",
                  11: "Z (id\_rho={:d}) (X={:.1E}) correct re-weighted iterative model",
                  12: "Z (id\_rho={:d}) (X={:.1E}) mangled stuff",
                  }

        # print Z values
        for idx_FS, FS in enumerate(self.FS_lst):
            for T in self.lst_T[idx_FS]:
                idx_T = self.lst_T[idx_FS].index(T)
                tau_values = generate_tau_values(self, T, idx_FS)
                # for X in self.lst_X[idx_FS]:  # don't plot more than the lowest # of samples
                # X = self.lst_X[idx_FS][0]
                X = int(1E4)
                idx_X = self.lst_X[idx_FS].index(X)

                x = tau_values.view()
                y = self.arr[idx_FS][:, idx_T, idx_X]["Z"].view()
                yerr = self.arr[idx_FS][:, idx_T, idx_X]["Z error"].view()
                label = labels[FS.id_rho].format(FS.id_rho, X)

                ax[idx_T].errorbar(x, y, xerr=None, yerr=yerr,
                                   marker='o' if idx_FS == 0 else 'x',
                                   markerfacecolor="None",
                                   label=label)
                ax[idx_T].legend()

                true_answer = self.sos_params[f"{T:.2f}"]["Z_coupled"]
                if self.percent_error:
                    true_answer = 0.0

                ax[idx_T].axhline(y=true_answer, linewidth=2, color='r',
                                  label='SOS derived Z from coupled model'
                                  )

        # Add an inset to the plot!
        if self.FS_lst[0].id_data >= 100:
            print("index A")
            from mpl_toolkits.axes_grid1.inset_locator import mark_inset
            from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

            # Build and manually place the inset (for the rho_1 case )

            # ax[0].xaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(11, 10)))

            ax_ins = fig.add_axes([0, 0, 1, 1], label='inset 1')
            ip = InsetPosition(ax[0], [0.1, 0.3, 0.5, 0.6])  # [%left, %up, %width, %height]
            ax_ins.set_axes_locator(ip)
            # Connect the inset to its respective region
            mark_inset(ax[0], ax_ins, loc1=3, loc2=4, fc="none", ec="0.5")

            # window dressing?
            # ax_ins.set_xscale('log')
            ax_ins.get_xaxis().set_visible(False)
            ax_ins.tick_params(which='both', direction='in', pad=4)
            # ax_ins.spines['top'].set_visible(False)

            # HACK CONSTANTS
            # F_index = -1
            idx_T = 0
            # idx_X = 0
            T = 300.00
            # dataView = self.arr[F_index, R_index, T_index, 0, :].view()
            slice_insert = np.s_[-1:-2:-1]
            # if np.any(np.isnan(dataView)):
            #     first_Finite = np.where(np.isfinite(self.arr[F_index, R_index, T_index, 0, :]))[0][-1]
            #     slice_insert = np.s_[first_Finite:first_Finite-10:-1]
            # --------------------------------------------------------------------------------
            # Plot the first FS
            print("index B")
            idx_FS = 0
            X = int(1E4)
            idx_X = self.lst_X[idx_FS].index(X)
            tau_values = self.generate_tau_values(T, idx_FS)
            x = tau_values[slice_insert].view()
            y = self.arr[idx_FS][slice_insert, idx_T, idx_X]["Z"].view()
            yerr = self.arr[idx_FS][slice_insert, idx_T, idx_X]["Z error"].view()
            label = labels[idx_FS].format(self.FS_lst[idx_FS].id_rho, self.lst_X[idx_FS][idx_X])
            ax_ins.errorbar(x, y, xerr=None, yerr=yerr,
                            marker='o' if idx_FS == 0 else 'x',
                            markerfacecolor="None",
                            label=label)
            # --------------------------------------------------------------------------------
            # Plot the second FS
            print(self.arr[idx_FS].shape)
            idx_FS = 1
            X = int(1E4)
            idx_X = self.lst_X[idx_FS].index(X)
            tau_values = self.generate_tau_values(T, idx_FS)
            print(self.lst_P[idx_FS])
            print("tau", tau_values)
            x = tau_values[slice_insert].view()
            y = self.arr[idx_FS][slice_insert, idx_T, idx_X]["Z"].view()
            yerr = self.arr[idx_FS][slice_insert, idx_T, idx_X]["Z error"].view()
            label = labels[idx_FS].format(self.FS_lst[idx_FS].id_rho, self.lst_X[idx_FS][idx_X])
            ax_ins.errorbar(x, y, xerr=None, yerr=yerr,
                            marker='o' if idx_FS == 0 else 'x',
                            markerfacecolor="None",
                            label=label)
            # --------------------------------------------------------------------------------
            # REFERENCE / EXACT ANSWER
            true_answer = self.sos_params[f"{T:.2f}"]["Z_coupled"]
            if self.percent_error:
                true_answer = 0.0

            ax_ins.axhline(y=true_answer, linewidth=2, color='r',
                           label='SOS derived Z from coupled model'
                           )
            # --------------------------------------------------------------------------------

        #
        x_label = r'$\tau\,(\text{eV}^{-1})$'
        ax[0].set_xlabel(x_label)
        ax[0].set_xscale('log')

        # plt.minorticks_off() # turns off minor ticks that are added with a log plot
        # points BOTH ticks inward
        ax[0].tick_params(which='both', direction='in', pad=2)

        # Remove the plot frame lines. They are unnecessary chart junk.
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)

        # y_label = r"$\frac{Z_g}{Z_\varrho}$"
        y_label = r"$Z_H$"
        if self.percent_error:
            # y_label = r"\% Difference    $\frac{Z_g}{Z_\varrho}$"
            y_label = r"\% Difference    $Z_H$"

        ax[0].set_ylabel(y_label)
        # ax[0].set_yscale('log')

        # plot title
        A, N = vIO.extract_dimensions_of_diagonal_model(self.FS_lst[0])
        plot_title = r'$\tau$ convergence'
        plot_title += f"\nData Set {self.FS_lst[0].id_data:d}"
        plot_title += f"\n{A:d} surfaces {N:d} normal modes"
        fig.suptitle(plot_title)

        fig.set_size_inches(10, 10)
        filename = "Z_sos_D{:d}_R{:d}_R{:d}.pdf".format(self.FS_lst[0].id_data, self.FS_lst[0].id_rho, self.FS_lst[1].id_rho)
        path_out = join(self.FS_lst[0].path_vib_plots, filename)
        fig.savefig(path_out, transparent=True, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)

        return

    def plot(self):
        """ x """

        # necessary for now
        prepare_mpl_rc_file(pretty_but_slow=True)
        load_latex_module_on_server()

        self.plot_Z()

        return


class plot_rectangle(plot_Z_multiple_FS):

    def __init__(self, list_of_FileStructure_objects):
        """ x """
        assert len(list_of_FileStructure_objects) > 1, " this subclass takes 2 or more FS"
        super().__init__(list_of_FileStructure_objects)

    def prepare_data(self):
        """ x """

        # we need to modify Z using the analytical rho (Z_rho)
        for idx_FS, FS in enumerate(self.FS_lst):
            for idx_T, T in enumerate(self.lst_T[idx_FS]):
                Z_rho = self.analytical_rho_list[idx_FS][f"{T:.2f}"]["Z_sampling"].view()
                self.arr[idx_FS][:, idx_T, :]["Z"] *= Z_rho
                self.arr[idx_FS][:, idx_T, :]["Z error"] *= Z_rho

        # we might also want to modify it to print the percent error
        self.percent_error = True

        if not self.percent_error:
            return

        for idx_FS, FS in enumerate(self.FS_lst):
            for X in self.lst_X[idx_FS]:  # don't plot more than the lowest # of samples
                idx_X = self.lst_X[idx_FS].index(X)
                for T in self.lst_T[idx_FS]:
                    idx_T = self.lst_T[idx_FS].index(T)

                    sos = self.sos_list[idx_FS][f"{T:.2f}"]
                    view = self.arr[idx_FS][:, idx_T, idx_X].view()

                    view["Z"] -= sos
                    view["Z"] *= 100.0
                    view["Z"] /= sos

                    view["Z error"] *= 100.0
                    view["Z error"] /= sos

                    # this isn't exactly what we want to do -- should change
                    trotter = self.trotter_coupled_list[idx_FS][:][f"{T:.2f}"]
                    trotter -= sos
                    trotter *= 100.0
                    trotter /= sos

                    # set the sos to zero since we are doing the percent option
                    sos[idx_FS][f"{T:.2f}"] = 0.0

        return

    def load_data(self):
        """ x """
        self.load_pimc_data()
        load_sos(self)
        load_trotter_coupled(self)
        load_analytical_sampling(self)
        self.prepare_data()
        return

    # TODO - this could be improved
    def generate_tau_values(self, temperature, idx_FS):
        """ returns a numpy array of the same length as lst_P
        takes in one temperature and an array of P values"""
        tau_arr = np.full(len(self.lst_P[idx_FS]), fill_value=beta(temperature))
        tau_arr /= self.lst_P[idx_FS]
        return tau_arr

    def plot_Z(self):
        # to be done later this week
        return


    def plot(self):
        """ x """

        # necessary for now
        prepare_mpl_rc_file(pretty_but_slow=True)
        load_latex_module_on_server()

        self.plot_Z()
        return


class plot_Z_test(plotVirtual):
    """ plotting when we only provide 1 FS object"""

    def generate_file_lists(self):
        """ create a list of lists of parameters specific to each FileStructure object """
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

            # bit of a naive way to load the data?
            # should possibly do some validation?
            with open(path, 'r') as file:
                data = json.loads(file.read())

            for key in data:
                if key != "hash_vib" and key != "hash_rho":
                    self.arr[idx_P, idx_T, idx_X][key] = data[key]
        return

    def plot_Z(self):
        fig, ax = plt.subplots(1, 1)

        # HACKY - this won't work anymore with the nested lists
        if len(self.lst_T[0]) is 1:
            ax = [ax, ]

        idx_FS = 0

        # print Z values
        for T in self.lst_T[idx_FS]:
            idx_T = self.lst_T[idx_FS].index(T)
            tau_values = generate_tau_values(self, T, idx_FS)
            for X in self.lst_X[idx_FS]:
                idx_X = self.lst_X[idx_FS].index(X)

                x = tau_values.view()
                y = self.arr[:, idx_T, idx_X]["Z"].view()
                yerr = self.arr[:, idx_T, idx_X]["Z error"].view()
                label = "Z (T={:.2f}K) (X={:.1E})".format(T, X)

                ax[idx_T].errorbar(x, y, xerr=None, yerr=yerr, marker='o', label=label)
                ax[idx_T].legend()

        fig.suptitle("Z MC")
        fig.set_size_inches(10, 10)
        filename = "Z_D{:d}_R{:d}.pdf".format(self.FS_lst[idx_FS].id_data,
                                              self.FS_lst[idx_FS].id_rho)
        path_out = join(self.FS_lst[idx_FS].path_rho_plots, filename)
        fig.savefig(path_out, transparent=True, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        return

    def plot_E(self):
        fig, ax = plt.subplots(1, 1)

        # HACKY - this won't work anymore with the nested lists
        if len(self.lst_T[0]) is 1:
            ax = [ax, ]

        idx_FS = 0

        # print E values
        for T in self.lst_T[idx_FS]:
            idx_T = self.lst_T[idx_FS].index(T)
            tau_values = generate_tau_values(self, T, idx_FS)
            for X in self.lst_X[idx_FS]:
                idx_X = self.lst_X[idx_FS].index(X)

                x = tau_values.view()
                y = self.arr[:, idx_T, idx_X]["E"].view()
                yerr = self.arr[:, idx_T, idx_X]["E error"].view()
                label = "E (T={:.2f}K) (X={:.1E})".format(T, X)

                ax[idx_T].errorbar(x, y, xerr=None, yerr=yerr, marker='o', label=label)
                ax[idx_T].legend()

        fig.suptitle("Energy")
        fig.set_size_inches(10, 10)
        filename = "E_D{:d}_R{:d}.pdf".format(self.FS_lst[idx_FS].id_data,
                                              self.FS_lst[idx_FS].id_rho)
        path_out = join(self.FS_lst[idx_FS].path_rho_plots, filename)
        fig.savefig(path_out, transparent=True, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        return

    def plot_Cv(self):
        fig, ax = plt.subplots(1, 1)

        # HACKY - this won't work anymore with the nested lists
        if len(self.lst_T[0]) is 1:
            ax = [ax, ]

        idx_FS = 0

        # print Cv values
        for T in self.lst_T[idx_FS]:
            idx_T = self.lst_T[idx_FS].index(T)
            tau_values = self.generate_tau_values(T)
            for X in self.lst_X[idx_FS]:
                idx_X = self.lst_X[idx_FS].index(X)

                x = tau_values.view()
                y = self.arr[:, idx_T, idx_X]["Cv"].view()
                yerr = self.arr[:, idx_T, idx_X]["Cv error"].view()
                label = "Cv (T={:.2f}K) (X={:.1E})".format(T, X)

                ax[idx_T].errorbar(x, y, xerr=None, yerr=yerr, marker='o', label=label)
                ax[idx_T].legend()

        fig.suptitle("Cv")
        fig.set_size_inches(10, 10)
        filename = "Cv_D{:d}_R{:d}.pdf".format(self.FS_lst[idx_FS].id_data,
                                               self.FS_lst[idx_FS].id_rho)
        path_out = join(self.FS_lst[idx_FS].path_rho_plots, filename)
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
