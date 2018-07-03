# setup_models.py - creates all the directories and copies in the necessary files

# system imports
import subprocess

# third party imports

# local imports
import context
import systems
# from pibronic import julia_wrapper
from pibronic import constants
from pibronic.log_conf import log
from pibronic.vibronic import vIO
from pibronic.server import job_boss
import pibronic.data.file_structure as fs


def generate_analytical_results(root, id_data, id_rho, temperature_list):
    """generate analytical results using julia"""
    systems.assert_id_data_is_valid(id_data)

    # TODO - should have a check to see if the analytical results already exists
    # and also should check that the hash values match

    # path_script = os.path.abspath(julia_wrapper.__file__)

    func_call = (
                'import pibronic;'
                'from pibronic.data import file_structure as fs;'
                'from pibronic import julia_wrapper as jw;'
                'from pibronic.constants import beta;'
                'b = beta({temperature:.2f});'
                f'FS = fs.FileStructure("{root:s}", {id_data:d}, {id_rho:d});'
                'FS.generate_model_hashes();'
                'jw.prepare_julia();'
                'pibronic.julia_wrapper.analytic_of_original_coupled_model(FS, b);'
                )

    for T in temperature_list:
        # this should also be generalized
        log.flow("About to generate analytical parameters at Temperature {:.2f}".format(T))

        cmd = ("srun"
               f" --job-name=analytc_D{id_data:d}_R{id_rho:d}_T{T:.2f}"
               " python3"
               " -c '{:s}'".format(func_call.format(temperature=T))
               )

        p = subprocess.Popen(cmd,
                             universal_newlines=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=True,
                             )
        # (out, error) = p.communicate()
        # print(out)
        # print(error)
    return


def simple_pimc_wrapper(root=None, id_data=11, id_rho=0):
    """ submits PIMC job to the server """
    systems.assert_id_data_is_valid(id_data)

    if root is None:
        root = context.choose_root_folder()

    # instantiate the FileStructure object which creates the directories
    FS = fs.FileStructure(root, id_data, id_rho)

    # possible bead values to choose from
    lst_P1 = [12, ]  # single testing case

    # two larger ranges
    lst_P2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, ]
    lst_P3 = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
              100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300]

    # don't resubmit the same jobs if not needed
    lst_P3 = list(set(lst_P3) - set(lst_P2))

    lst_P3.sort()

    lst_T = [300.00, ]
    # lst_T = [250., 275., 300., 325., 350., ]

    # TODO - need to add a check that only generates analytical results if the
    # file doesn't eists OR the hash values don't match up!
    generate_analytical_results(FS.path_root, id_data, id_rho, lst_T)
    A, N = vIO.extract_dimensions_of_coupled_model(FS=FS)

    # this is the minimum amount of data needed to run an execution
    parameter_dictionary = {
        "number_of_samples": int(1e5),
        "number_of_states": A,
        "number_of_modes": N,
        "bead_list": lst_P2,
        "temperature_list": lst_T,
        "delta_beta": constants.delta_beta,
        "id_data": id_data,
        "id_rho": id_rho,
    }

    # create an execution object
    engine = job_boss.PimcSubmissionClass(FS, parameter_dictionary)

    engine.submit_jobs()

    return


def automate_wrapper(name):
    """ loops over the data sets and different rhos submiting PIMC jobs for each one  """
    systems.assert_system_name_is_valid(name)

    for id_data in systems.id_dict[name]:
        for id_rho in systems.rho_dict[name][id_data]:
            simple_pimc_wrapper(id_data=id_data, id_rho=id_rho)
    return


if (__name__ == "__main__"):

    # eventual code
    # map(automate_wrapper, systems.name_lst)

    # during testing
    automate_wrapper(systems.name_lst[0])
    automate_wrapper(systems.name_lst[1])
    automate_wrapper(systems.name_lst[2])
