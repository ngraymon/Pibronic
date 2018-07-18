# separating out the basic plotting class structure

# system imports

# third party imports

# local imports
from ..data import file_structure as fs


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
        # might want to call these functions separately from initialization
        # self.validate_data()  # this one will be tricky - might be optional
        # prepare_mpl_rc_file()
        return
