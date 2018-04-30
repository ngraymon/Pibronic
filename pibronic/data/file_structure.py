"""
Bookkeeping provided by FileStucture class

keeps directory structure consistant over the many modules if changes need to be made
"""

# system imports
import os
import sys

# third party imports
import numpy as np

# local imports
from . import file_name
from .. import log_conf


# the names of the sub directories
# TODO - clean this up
list_sub_dirs = [
    "parameters/",
    "results/",
    "execution_output/",
    "plots/",
    ]


class FileStructure:
    """handles creation and verification of default file structure storing data from/to jobs"""
    template_vib = "data_set_{:d}/"
    template_rho = template_vib + "rho_{:d}/"
    template_vib_params = template_vib + "parameters/"
    template_vib_results = template_vib + "results/"
    template_vib_output = template_vib + "execution_output/"
    template_vib_plots = template_vib + "plots/"
    template_rho_params = template_rho + "parameters/"
    template_rho_results = template_rho + "results/"
    template_rho_output = template_rho + "execution_output/"
    template_rho_plots = template_rho + "plots/"

    # TODO - make sure all the old output files are not needed or have been converted to new file names
    """
    template_pimc_suffix = "D{D:d}_R{R:d}_P{P:d}_T{T:.2f}_J*_data_points.npz"
    template_jackknife_suffix = "D{D:d}_R{R:d}_P{P:d}_T{T:.2f}_X{X:d}_thermo"
    template_sos_suffix = "sos_B{B:d}.json"
    """

    # template_pimc_suffix = file_name.pimc
    # template_jackknife_suffix = file_name.jackknife
    # template_sos_suffix = file_name.sos

    @classmethod
    def from_boxdata(cls, path_root, data):
        """constructor wrapper that takes a data object from minimal.py"""
        return cls(path_root, data.id_data, data.id_rho)

    def __init__(self, path_root, id_data, id_rho=0):
        """x"""
        assert(type(path_root) == str)
        self.id_rho = id_rho
        self.id_data = id_data
        self.path_root = path_root
        self.path_data = path_root + self.template_vib.format(id_data)
        self.path_rho = path_root + self.template_rho.format(id_data, id_rho)
        # special
        self.path_es = self.path_data + "electronic_structure/"

        self.path_vib_params = path_root + self.template_vib_params.format(id_data)
        self.path_vib_results = path_root + self.template_vib_results.format(id_data)
        self.path_vib_output = path_root + self.template_vib_output.format(id_data)
        self.path_vib_plots = path_root + self.template_vib_plots.format(id_data)
        self.path_rho_params = path_root + self.template_rho_params.format(id_data, id_rho)
        self.path_rho_results = path_root + self.template_rho_results.format(id_data, id_rho)
        self.path_rho_output = path_root + self.template_rho_output.format(id_data, id_rho)
        self.path_rho_plots = path_root + self.template_rho_plots.format(id_data, id_rho)

        # root_suffix = "D{:d}_R{:d}_".format(id_data, id_rho)
        # self.pimc_suffix = root_suffix + "P{P:d}_T{T:.2f}_J*_data_points.npz"
        # self.jackknife_suffix = root_suffix + "P{P:d}_T{T:.2f}_X{X:d}_thermo"

        self.template_pimc = self.path_rho_results + file_name.pimc
        self.template_jackknife = self.path_rho_results + file_name.jackknife
        self.template_sos_rho = self.path_rho_params + file_name.sos
        self.template_sos_vib = self.path_vib_params + file_name.sos

        # TODO - should we factor this out into the file_name module?
        # self.template_pimc = file_name.pimc
        # self.pimc_suffix = "P{P:d}_T{T:.2f}_J*_data_points.npz"
        # self.jackknife_suffix = self.template_jackknife_suffix

        self.dir_list = [a for a in dir(self) if a.startswith('path_')]
        # print(self.dir_list, '\n\n')
        # self.dir_list = [self.path_data, self.path_es]
        # self.dir_list.extend([self.path_data + x for x in list_sub_dirs])
        # self.dir_list.extend([self.path_rho + x for x in list_sub_dirs])

        """ TODO - possibly rename the sampling and coupled paths?
        they are paths but shouldn't be in the dir_list
        """
        self.path_vib_model = self.path_vib_params + file_name.coupled_model
        self.path_har_model = self.path_vib_params + file_name.harmonic_model
        self.path_rho_model = self.path_rho_params + file_name.sampling_model

        # if not self.directories_exist():
        # self.make_directories()
        return

    def directories_exist(self):
        """x"""
        if False in map(os.path.isdir, self.dir_list):
            return False
        return True

    def make_directories(self):
        """x"""
        for directory in self.dir_list:
            os.makedirs(directory, exist_ok=True)
        return

    def make_rho_directories(self, id_rho):
        """x"""
        if id_rho == self.id_rho:
            self.make_directories()
            return

        directories = [self.path_data + self.dir_rho.format(id_rho) + x for x in list_sub_dirs]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        return

    def verify_directories_exists(self, id_data, path_root=None):
        """ if directory structure exists does nothing, creates the file structure otherwise"""

        # assume default
        if path_root is None:
            raise Exception("NO DEFAULT ROOT!?")
            # path_root = path_default_root # fix this ?

        files = FileStructure(path_root, id_data)
        if files.directories_exist():
            return

        files.make_directories()
        return
