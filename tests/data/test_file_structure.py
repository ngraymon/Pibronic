""" """

# system imports
import os
from os.path import join
from os.path import normpath
from os.path import samefile

# local imports
from ..context import pibronic
import pibronic.data.file_structure as fs

# third party imports
import pytest


def test_FileStructure_directory_creation(tmpdir):
    root = tmpdir.mkdir("root")
    id_data = 0
    id_rho = 0
    FS = fs.FileStructure(root, id_data, id_rho)

    dict_path = {
                 "path_root": "",
                 "path_data": f"data_set_{id_data:}",
                 "path_es": f"data_set_{id_data:}/electronic_structure",
                 "path_vib_params": f"data_set_{id_data:}/parameters",
                 "path_vib_results": f"data_set_{id_data:}/results",
                 "path_vib_output": f"data_set_{id_data:}/execution_output",
                 "path_vib_plots": f"data_set_{id_data:}/plots",
                 "path_rho": f"data_set_{id_data:}/rho_{id_rho}",
                 "path_rho_params": f"data_set_{id_data:}/rho_{id_rho}/parameters",
                 "path_rho_results": f"data_set_{id_data:}/rho_{id_rho}/results",
                 "path_rho_output": f"data_set_{id_data:}/rho_{id_rho}/execution_output",
                 "path_rho_plots": f"data_set_{id_data:}/rho_{id_rho}/plots",
                 }

    for key, value in dict_path.items():
        assert samefile(getattr(FS, key), join(root, value))

    return
