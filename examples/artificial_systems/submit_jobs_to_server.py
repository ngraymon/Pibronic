#

# system imports
from multiprocessing import Pool
import subprocess
import json
from os.path import isfile

# third party imports

# local imports
import context
import systems
from pibronic import constants
from pibronic.log_conf import log
from pibronic.vibronic import vIO
from pibronic.server import job_boss
import pibronic.data.file_structure as fs

jl_import = (
          'import pibronic;'
          'from pibronic.data import file_structure as fs;'
          'from pibronic import julia_wrapper as jw;'
          'from pibronic.constants import beta;'
           )


def _valid_hash(FS, path):
    """ make sure that the file contains the correct hash value"""
    # path = FS.template_sos_vib.format(B=basis_size)
    with open(path, 'r') as file:
        data = json.loads(file.read())
    return FS.valid_vib_hash(data)


def _temp_already_present(FS, path, temp):
    """ return true if that temperature value is already present in the dictionary"""
    if isfile(path):
        with open(path, 'r') as file:
            data = json.loads(file.read())
        return bool(f"{temp:.2f}" in data.keys())
    return False


def _generate_sos_results(FS, temperature_list, basis_size_list):
    """ contains the plumbing to submit a job to Slurm which runs the Julia script to calculate sos parameters for the model of interest that was identified by the input parameters """
    func_call = jl_import
    func_call += (
                   'b = beta({temperature:.2f});'
                   'BS = {BS:d};'
                   f'FS = fs.FileStructure("{FS.path_root:s}", {FS.id_data:d}, {FS.id_rho:d});'
                   'FS.generate_model_hashes();'
                   'jw.prepare_julia();'
                   'pibronic.julia_wrapper.sos_of_coupled_model(FS, BS, b);'
                   )

    for BS in basis_size_list:
        for T in temperature_list:

            path = FS.template_sos_vib.format(B=BS)
            if isfile(path) and _valid_hash(FS, path) and _temp_already_present(FS, path, T):
                continue

            log.flow(f"About to generate sos parameters at Temperature {T:.2f}")

            cmd = ("srun"
                   f" --job-name=sos_D{FS.id_data:d}_T{T:.2f}"
                   # " --cpus-per-task=1"  # default Julia shows no speed ups for multi-core
                   " --mem=20GB"
                   " python3 -c '{:s}'".format(func_call.format(temperature=T, BS=BS))
                   )

            p = subprocess.Popen(cmd, shell=True, universal_newlines=True,
                                 # stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 )
            # (out, error) = p.communicate()

            # if error != '' and "Out Of Memory" in error:
            #     print(error, "\nMemory error, you need to run this job on toby\n")
            #     print(out)
            #     exit(0)
            #     continue
            # print(error)
            # print(out)
    return


def _generate_trotter_results(FS, temperature_list, basis_size_list, bead_list):
    """ contains the plumbing to submit a job to Slurm which runs the Julia script to calculate trotter parameters for the model of interest that was identified by the input parameters """
    func_call = jl_import
    func_call += (
                   'b = beta({temperature:.2f});'
                   'P = {nbeads:d};'
                   'BS = {BS:d};'
                   f'FS = fs.FileStructure("{FS.path_root:s}", {FS.id_data:d}, {FS.id_rho:d});'
                   'FS.generate_model_hashes();'
                   'jw.prepare_julia();'
                   'pibronic.julia_wrapper.trotter_of_coupled_model(FS, P, BS, b);'
                   )

    for BS in basis_size_list:
        for T in temperature_list:
            for P in bead_list:

                path = FS.template_trotter_vib.format(P=P, B=BS)
                if isfile(path) and _valid_hash(FS, path) and _temp_already_present(FS, path, T):
                    continue

                log.flow(f"About to generate trotter parameters at Temperature {T:.2f}")

                cmd = ("srun"
                       f" --job-name=trotter_D{FS.id_data:d}_P{P:d}_T{T:.2f}"
                       # " --cpus-per-task=1"  # default Julia shows no speed ups for multi-core
                       " --mem=20GB"
                       " python3 -c '{:s}'".format(func_call.format(temperature=T, nbeads=P, BS=BS))
                       )

                p = subprocess.Popen(cmd, shell=True, universal_newlines=True,
                                     # stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     )
                # (out, error) = p.communicate()

                # if error != '' and "Out Of Memory" in error:
                #     print(error, "\nMemory error, you need to run this job on toby\n")
                #     continue
                # print(error)
                # print(out)
    return


def simple_sos_wrapper(lst_BS, lst_T, root=None, id_data=11):
    """ submits sos jobs to the server """
    systems.assert_id_data_is_valid(id_data)

    if root is None:
        root = context.choose_root_folder()

    FS = fs.FileStructure(root, id_data, id_rho=0)
    FS.generate_model_hashes()

    _generate_sos_results(FS, lst_T, lst_BS)
    return


def simple_trotter_wrapper(lst_BS, lst_T, lst_P, root=None, id_data=11):
    """ submits trotter jobs to the server """
    systems.assert_id_data_is_valid(id_data)

    if root is None:
        root = context.choose_root_folder()

    FS = fs.FileStructure(root, id_data, id_rho=0)
    FS.generate_model_hashes()

    _generate_trotter_results(FS, lst_T, lst_BS, lst_P)
    return


def automate_sos_trotter_submission(name):
    """ loops over the data sets and different rhos submitting sos and trotter jobs for each one  """
    systems.assert_system_name_is_valid(name)

    assert name is "2x2", "Any other model is too large to do trotter or sos"

    lst_BS = [80, ]
    lst_T = [300.00, ]

    lst_P1 = [10, ]
    lst_P2 = [10, 50, 100, 150, 200, ]
    lst_P3 = [50, 100, 150, 200, 300, 400, 500, ]

    for id_data in systems.id_dict[name]:
        simple_sos_wrapper(lst_BS, lst_T, id_data=id_data)
        simple_trotter_wrapper(lst_BS, lst_T, lst_P3, id_data=id_data)
    return


def iterative_method_wrapper(root=None, id_data=11):
    """ submits iterative Julia script to the server """
    systems.assert_id_data_is_valid(id_data)

    if root is None:
        root = context.choose_root_folder()

    # instantiate the FileStructure object which creates the directories
    FS = fs.FileStructure(root, id_data, id_rho=0)

    number_of_iterations = 50

    func_call = jl_import
    func_call += (
                   f'n_iter = {number_of_iterations:d};'
                   f'FS = fs.FileStructure("{FS.path_root:s}", {FS.id_data:d}, {FS.id_rho:d});'
                   'jw.prepare_julia();'
                   'pibronic.julia_wrapper.iterate_method(FS, n_iterations=n_iter);'
                   )

    log.flow("About to generate iterative sampling distribution")

    # TODO - adjust the run time parameters
    cmd = ("srun"
           " --pty"
           f" --job-name=iterative_D{FS.id_data:d}"
           # " --cpus-per-task=1"  # default julia hasn't seen speed ups for more than 1 core
           " --mem=20GB"
           " python3 -c '{:s}'".format(func_call)
           )

    p = subprocess.Popen(cmd, shell=True, universal_newlines=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         )
    (out, error) = p.communicate()

    if error != '' and "Out Of Memory" in error:
        print(error, "\nMemory error, you need to run this job on toby\n")

    print(out)

    return


def automate_iterative_submission(name):
    """ loops over the data sets and different rhos submitting PIMC jobs for each one  """
    systems.assert_system_name_is_valid(name)

    for id_data in systems.id_dict[name]:
        iterative_method_wrapper(id_data=id_data)
    return


def _generate_original_analytic_results(FS, temperature_list):
    """ wrapper for convenience """
    func_call = jl_import
    func_call += (
                  'b = beta({temperature:.2f});'
                  f'FS = fs.FileStructure("{FS.path_root:s}", {FS.id_data:d}, {FS.id_rho:d});'
                  'FS.generate_model_hashes();'
                  'jw.prepare_julia();'
                  'pibronic.julia_wrapper.analytic_of_coupled_model(FS, b);'
                  )

    for T in temperature_list:
        log.flow(f"About to generate analytical parameters of original coupled model at Temperature {T:.2f}")

        cmd = ("srun"
               f" --job-name=analytic_D{FS.id_data:d}_T{T:.2f}"
               " python3 -c '{:s}'".format(func_call.format(temperature=T))
               )

        p = subprocess.Popen(cmd, shell=True, universal_newlines=True,
                             # stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             )
        # (out, error) = p.communicate()
        # print(out)
        # print(error)
    return


def _generate_sampling_analytic_results(FS, temperature_list):
    """ wrapper for convenience """
    func_call = jl_import
    func_call += (
                   'b = beta({temperature:.2f});'
                   f'FS = fs.FileStructure("{FS.path_root:s}", {FS.id_data:d}, {FS.id_rho:d});'
                   'FS.generate_model_hashes();'
                   'jw.prepare_julia();'
                   'pibronic.julia_wrapper.analytic_of_sampling_model(FS, b);'
                   )

    for T in temperature_list:
        log.flow(f"About to generate analytical parameters of diagonal model at Temperature {T:.2f}")

        cmd = ("srun"
               f" --job-name=analytic_D{FS.id_data:d}_R{FS.id_rho:d}_T{T:.2f}"
               " python3 -c '{:s}'".format(func_call.format(temperature=T))
               )

        p = subprocess.Popen(cmd, shell=True, universal_newlines=True,
                             # stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             )
        # (out, error) = p.communicate()
        # print(out)
        # print(error)
    return


def generate_analytical_results(FS, temperature_list):
    """generate analytical results using Julia"""
    systems.assert_id_data_is_valid(FS.id_data)

    # TODO - should have a check to see if the analytical results already exists
    # and also should check that the hash values match
    _generate_original_analytic_results(FS, temperature_list)
    _generate_sampling_analytic_results(FS, temperature_list)


def simple_pimc_wrapper(root=None, id_data=11, id_rho=0):
    """ submits PIMC job to the server """
    systems.assert_id_data_is_valid(id_data)

    if root is None:
        root = context.choose_root_folder()

    # instantiate the FileStructure object which creates the directories
    FS = fs.FileStructure(root, id_data, id_rho)

    # possible bead values to choose from
    lst_P1 = [12, ]  # single testing case

    # use logspace to generate bead values to get even spacing on plot
    # two larger ranges
    lst_P2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, ]
    lst_P3 = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
              100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300]

    lst_P4 = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]
    lst_P5 = [50, 100, 150, 200, 300, 400, 500]
    # don't resubmit the same jobs if not needed
    # lst_P3 = list(set(lst_P3) - set(lst_P2))

    lst_P3.sort()

    lst_T = [300.00, ]
    # lst_T = [250., 275., 300., 325., 350., ]

    # TODO - need to add a check that only generates analytical results if the
    # file doesn't exists OR the hash values don't match up!
    generate_analytical_results(FS, lst_T)
    A, N = vIO.extract_dimensions_of_model(FS=FS)
    # this is the minimum amount of data needed to run an execution
    parameter_dictionary = {
        "number_of_samples": int(1e5),
        "number_of_states": A,
        "number_of_modes": N,
        "bead_list": lst_P5,
        "temperature_list": lst_T,
        "delta_beta": constants.delta_beta,
        "id_data": id_data,
        "id_rho": id_rho,
    }

    # create an execution object
    engine = job_boss.PimcSubmissionClass(FS, parameter_dictionary)

    engine.submit_jobs()

    return


def automate_pimc_submission(name):
    """ loops over the data sets and different rhos submitting PIMC jobs for each one  """
    systems.assert_system_name_is_valid(name)

    for id_data in systems.id_dict[name]:
        for id_rho in systems.rho_dict[name][id_data]:
            simple_pimc_wrapper(id_data=id_data, id_rho=id_rho)
    return


def main():
    # If you want to speed things up you can split the work across four processes
    multiprocessing_flag = False

    if multiprocessing_flag:
        with Pool(len(systems.name_lst)) as p:
                p.map(automate_pimc_submission, systems.name_lst)
                # p.map(automate_sos_trotter_submission, systems.name_lst)
    else:
        # during testing
        # Sequential, comment out lines if you only need to run for individual models
        automate_pimc_submission(systems.name_lst[0])
        # automate_pimc_submission(systems.name_lst[1])
        # automate_pimc_submission(systems.name_lst[2])
        # automate_sos_trotter_submission(systems.name_lst[0])
        # automate_sos_trotter_submission(systems.name_lst[1])
        # automate_sos_trotter_submission(systems.name_lst[2])
        pass


if (__name__ == "__main__"):
    main()
