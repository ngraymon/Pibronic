""" analysis plotting - intensity grid of the parameters of a model"""


# system imports
import json
from os.path import join
import itertools as it

# third party imports
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt

# local imports
from .server import prepare_mpl_rc_file, load_latex_module_on_server
from ..data import file_structure as fs
from ..vibronic import vIO, VMK

# a useful resource for plotting the heatmaps/grids
# https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py

# a simple reminder of modifying matplotlib ticks
# https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html


class plotGrids():

    def __init__(self, input_FS):
        """ x """
        assert isinstance(input_FS, fs.FileStructure), "this class only takes 1 FS"
        self.FS = input_FS
        return

    def load_original_models(self):
        """ x """
        path = self.FS.path_orig_model
        with open(path, 'r') as file:
            self.original_model = json.loads(file.read())
        return

    def load_coupled_models(self):
        """ x """
        path = self.FS.path_vib_model
        with open(path, 'r') as file:
            self.coupled_model = json.loads(file.read())
        return

    def prepare_data(self):
        """ optionally - if we want to modify the data before plotting """
        # self.original_data = np.array(self.original_model[VMK.E.value])
        # self.coupled_data = np.array(self.coupled_model[VMK.E.value])
        return

    def load_data(self):
        """ x """
        self.load_original_models()
        self.load_coupled_models()
        self.prepare_data()
        return

    def annotate_intensity_grid(self, fig, ax, data, title):
        """ commonly shared code for plotting"""
        A, N = vIO.extract_dimensions_of_coupled_model(self.FS)

        # We want 1 tick for each square
        plt.locator_params(nbins=A)
        ax.set_yticklabels(range(0, A+2))
        ax.set_xticklabels(np.arange(0, A+2))

        # Loop over data dimensions and create text annotations.
        annotations = self.generate_sign_array(data)
        for a, b in it.product(range(A), range(A)):
            text = ax.text(a, b, annotations[a, b], fontsize=50,
                           ha="center", va="center", color="k")

        label = "Electronic Surfaces"
        ax.set_ylabel(label)
        ax.set_xlabel(label)

        # Let the horizontal axes labeling appear on top.
        # ax.tick_params(top=False, bottom=False, labeltop=True, labelbottom=False)

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        # plot title
        plot_title = title
        plot_title += f"\nData Set {self.FS.id_data:d} - {A:d} surfaces {N:d} normal modes"
        fig.suptitle(plot_title)
        return

    def generate_sign_array(self, array):
        sign = np.sign(array)
        sign_array = np.full(shape=array.shape, fill_value='y', dtype=object)
        sign_array[np.where(sign == 1.)] = '+'
        sign_array[np.where(sign == -1.)] = '-'
        sign_array[np.where(sign == 0.)] = '0'
        return sign_array

    def plot_single_energy_grid(self, energy, filename, title):
        """ x """
        fig, ax = plt.subplots()

        im = ax.imshow(energy, origin='upper')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r'Energy values (eV)', rotation=-90, va="bottom")
        self.annotate_intensity_grid(fig, ax, energy, title)

        path_out = join(self.FS.path_vib_plots, filename)
        fig.savefig(path_out, transparent=True, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        return

    def plot_original_energy_grid(self):
        """ x """
        title = 'Original Energy Grid'
        filename = "original_energy_grid_D{:d}.pdf".format(self.FS.id_data)
        energy = np.array(self.original_model[VMK.E.value])
        self.plot_single_energy_grid(energy, filename, title)
        return

    def plot_coupled_energy_grid(self):
        """ x """
        title = 'Coupled Energy Grid'
        filename = "coupled_energy_grid_D{:d}.pdf".format(self.FS.id_data)
        energy = np.array(self.coupled_model[VMK.E.value])
        self.plot_single_energy_grid(energy, filename, title)
        return

    def plot_energy(self):
        """ x """
        # necessary for now
        prepare_mpl_rc_file(pretty_but_slow=True)
        load_latex_module_on_server()

        self.plot_original_energy_grid()
        self.plot_coupled_energy_grid()
        return

    def plot_single_linear_grid(self, energy, filename, title):
        """ x """
        fig, ax = plt.subplots()

        im = ax.imshow(energy, origin='upper')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r'Energy values (eV)', rotation=-90, va="bottom")
        self.annotate_intensity_grid(fig, ax, energy, title)

        path_out = join(self.FS.path_vib_plots, filename)
        fig.savefig(path_out, transparent=True, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        return

    def plot_original_linear_grids(self, data):
        """ x """
        A, N = vIO.extract_dimensions_of_coupled_model(self.FS)
        for j in range(N):
            title = f"Original linear terms Grid (Mode {j+1:d})"
            filename = f"original_linear_grid_D{self.FS.id_data:d}_N{j+1:d}.pdf"
            self.plot_single_linear_grid(data[j, ...], filename, title)
        return

    def plot_coupled_linear_grids(self, data):
        """ x """
        A, N = vIO.extract_dimensions_of_coupled_model(self.FS)
        for j in range(N):
            title = f"Coupled linear terms Grid (Mode {j+1:d})"
            filename = f"coupled_linear_grid_D{self.FS.id_data:d}_N{j+1:d}.pdf"
            self.plot_single_linear_grid(data[j, ...], filename, title)
        return

    def plot_linear_coupling(self):
        """ x """
        # necessary for now
        prepare_mpl_rc_file(pretty_but_slow=True)
        load_latex_module_on_server()

        if VMK.G1.value not in self.coupled_model:
            raise Exception("Linear terms are zero, nothing to plot")

        linear = np.array(self.coupled_model[VMK.G1.value])

        self.plot_original_linear_grids(linear)
        self.plot_coupled_linear_grids(linear)
        return

    def plot_single_quadratic_grid(self, energy, filename, title):
        """ x """
        fig, ax = plt.subplots()

        im = ax.imshow(energy, origin='upper')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r'Energy values (eV)', rotation=-90, va="bottom")
        self.annotate_intensity_grid(fig, ax, energy, title)

        path_out = join(self.FS.path_vib_plots, filename)
        fig.savefig(path_out, transparent=True, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        return

    def plot_original_quadratic_grids(self, data):
        """ x """
        A, N = vIO.extract_dimensions_of_coupled_model(self.FS)
        for j1,  j2 in it.product(range(N), range(N)):
            title = f"Original quadratic terms Grid (Modes {j1+1:d}{j2+1:d})"
            filename = f"original_quadratic_grid_D{self.FS.id_data:d}_N{j1+1:d}_N{j2+1:d}.pdf"
            self.plot_single_quadratic_grid(data[j1, j2, ...], filename, title)
        return

    def plot_coupled_quadratic_grids(self, data):
        """ x """
        A, N = vIO.extract_dimensions_of_coupled_model(self.FS)
        for j1,  j2 in it.product(range(N), range(N)):
            title = f"Coupled quadratic terms Grid (Modes {j1+1:d}{j2+1:d})"
            filename = f"coupled_quadratic_grid_D{self.FS.id_data:d}_N{j1+1:d}_N{j2+1:d}.pdf"
            self.plot_single_quadratic_grid(data[j1, j2, ...], filename, title)
        return

    def plot_quadratic_coupling(self):
        """ x """
        # necessary for now
        prepare_mpl_rc_file(pretty_but_slow=True)
        load_latex_module_on_server()

        if VMK.G2.value not in self.coupled_model:
            raise Exception("Linear terms are zero, nothing to plot")

        quadratic = np.array(self.coupled_model[VMK.G2.value])

        self.plot_original_quadratic_grids(quadratic)
        self.plot_coupled_quadratic_grids(quadratic)
        return
