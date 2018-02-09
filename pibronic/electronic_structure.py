# electronic_structure.py
#
# system imports
import itertools as it
import subprocess
import threading
import warnings
import inspect
import shutil
import socket
import signal
import enum
import time
import math
import mmap
import sys
import os

# third party imports
import numpy as np

# local imports
# from ..data import vibronic_model_io as vIO
# from .. import constants
# from ..constants import hbar
from .log_conf import log
from .server import job_boss

# how we identify the different steps
@enum.unique
class State(enum.Enum):
    # DROPMO   = 0
    OPT      = enum.auto()
    NIP      = enum.auto()
    VIB      = enum.auto()
    IPEOMCC  = enum.auto()
    PREPVIB  = enum.auto()
    VIBRON   = enum.auto()
    FINISHED = enum.auto()

    @classmethod
    def max(cls):
        return max([x.value for x in cls.__members__.values()])

    @classmethod
    def min(cls):
        return min([x.value for x in cls.__members__.values()])


# the input filename templates that this script uses to run calculations
default_file_in = {
    "parameter" : "{:s}_params.txt",
    "zmat"      : "{:s}_zmat.txt",
    "DROPMO"    : "{:s}_hartree_fock_dropmo.in",
    "NIP"       : "{:s}_hartree_fock_nip.in",
    "geometry"  : "{:s}_opt.in",
    "vib"       : "{:s}_vib.in",
    "ipeomcc"   : "{:s}_eomcc_ip_{{:d}}.in",
    "prepVib"   : "{:s}_prepvib.in",
    "vibronIn"  : "{:s}_vibron.in",
    # "vibronIn"  : "cp.in",
    "vibronCp"  : "cp.auto",
    # "vibronCp"  : "{:s}_vibron.auto",
}

# the output filename templates that this script uses to run calculations
default_file_out = {
    "DROPMO"    : "{:s}_hartree_fock_dropmo.out0",
    "NIP"       : "{:s}_hartree_fock_nip.out0",
    "geometry"  : "{:s}_opt.out0",
    "vib"       : "{:s}_vib.out0",
    "ipeomcc"   : "{:s}_eomcc_ip_{{:d}}.out0",
    "prepVib"   : "{:s}_prepvib.out0",
    "pntheff"   : "{:s}_prepvib.pntheff",
    "vibronIn"  : "{:s}_prepvib.vibron_input",
    "vibronCp"  : "{:s}_prepvib.vibronic_coupling",
    "vibron"    : "{:s}_vibron.out0",
    # "vibron"    : "cp.auto0",
}


# supported calculation methods
basis_list = [
    "TZ2P",
    "DZP",
    # "6-31+G",
    # "6-31++G**",
]

theory_list = [
    "MBPT(2)",
    "CCSD",
    "CCSD(T)",
]

calculations_list = [
    "IP",
    # "EA",
    # "EA",
]

# get the functions name
#inspect.currentframe().f_code.co_name
# get the name of the function that called the current function
#inspect.currentframe().f_back.f_code.co_name


# quick hacky cheats!!

file_name_state = "execution_state.txt"

# quick hacky cheats!!

# -----------------------------------------------------------
# MEMORY MAPPED HELPER FUNCTIONS
# -----------------------------------------------------------

def find_string_in_file(memmap_file, file_path, target_string):
    """wrapper that raises error if no substr can be found finds the first occurance of a substring in memory mapped file"""
    location = memmap_file.find(target_string.encode(encoding="utf-8"))

    if location == -1:
        # couldn't find target string in file
        s = (   "It seems \"{:s}\" was not present in the file\n\"{:s}\"\n"
                "Check that the previous calculation didn't fail"
            )
        raise Exception(s.format(target_string, file_path))
    return location


def rfind_string_in_file(memmap_file, file_path, target_string):
    """wrapper that raises error if no substr can be found finds the last occurance of a substring in memory mapped file"""
    location = memmap_file.rfind(target_string.encode(encoding="utf-8"))

    if location == -1:
        # couldn't find target string in file
        s = (   "It seems \"{:s}\" was not present in the file\n\"{:s}\"\n"
                "Check that the previous calculation didn't fail"
            )
        raise Exception(s.format(target_string, file_path))
    return location


def skip_back_n_lines(memmap_file, n, start_index):
    """gives the byte location n lines before the given byte location start_index"""
    for _ in it.repeat(None, n):
        temporary_start_index = memmap_file.rfind(b'\n', 0, start_index)
        if temporary_start_index < 0:
            break
        start_index = temporary_start_index

    return start_index


def skip_forward_n_lines(memmap_file, n, start_index):
    """gives the byte location n lines after the given byte location start_index"""
    for _ in it.repeat(None, n):
        temporary_start_index = memmap_file.find(b'\n', start_index + 1)
        if temporary_start_index == -1:
            break
        start_index = temporary_start_index

    return start_index

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------

def pretty_print_job_status():
    """ quick hack for scripting"""
    number_dictionary = {
        0   :   ("acetonitrile", 200),
        1   :   ("ammonia", 201),
        2   :   ("boron_trifluoride", 202),
        3   :   ("formaldehyde", 203),
        4   :   ("methane", 204),
        5   :   ("formamide", 205),
        6   :   ("formic_acid", 206),
        7   :   ("hydrogen_peroxide", 207),
        8   :   ("water", 208),
        9   :   ("pyridine", 209),
        10  :   ("furan", 210),
        11  :   ("trichloroethylene", 211),
        12  :   ("chloroethylene", 212),
        13  :   ("acrolein", 213),
        14  :   ("acrylonitrile", 214),
        15  :   ("cis_12_dichloroethylene", 215),
        16  :   ("trans_12_dichloroethylene", 216),
        17  :   ("11_dichloroethylene", 217),
        # 18  :   ("", 218),
        # 19  :   ("", 219),
        # 20  :   ("", 220),
        # 21  :   ("", 221),
        }

    path_root = "/work/ngraymon/pimc/data_set_{:d}/electronic_structure/execution_state.txt"

    for model in number_dictionary.values():
        path_file_state = path_root.format(model[1])
        if os.path.isfile(path_file_state):
            with open(path_file_state, 'r') as file:
                lines = file.readlines()
            for line in reversed(lines):
                if line in ["", "\n"]:
                    continue
                last_state = str(line.strip())
                break
            log.info("({:}, {:}) last state was {:}".format(model[0],model[1],last_state))
        else:
            s = "({:}, {:}) does not have an execution_state.txt file"
            log.info(s.format(model[0],model[1]))

    return


def verify_file_exists(file_path):
    """raises FileNotFoundError exception if input is not a file, or does not exist"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError("Cannot find {}".format(file_path))
    return True


def pairwise(iterable):
    """returns an iterator over pairs of elements in the iterable"""
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def get_yes_no_from_user(message):
    """return the boolean value of the users response the a yes/no question"""
    while True:
        try:
            yes_no = bool('y' in input(message).lower())
        except ValueError:
            print("Sorry, I didn't understand that\n")
            continue
        else:
            break

    return yes_no


def get_integer_from_user(message):
    """return the integer value of the users response to a input prompt"""
    while True:
        try:
            user_integer = int(input(message))
            assert(user_integer >= 0)
        except ValueError:
            print("Sorry, I didn't understand that\n")
            continue
        except AssertionError:
            print("Please provide a non negative value\n")
            continue
        else:
            break

    return user_integer


def wait_on_results(job_id, job_type):
    """sychronizes with the submitted job and verifies that the job's output is successful"""
    log.flow("Waiting on job")

    # wait for the job to queue up
    time.sleep(5)

    job_boss.synchronize_with_job(job_id, job_type)

    # get the recorded state of the job
    result = job_boss.check_acct_state(job_id)

    # check if the job failed
    if "FAILED" in result:
        s = "{:s} script (JOBID={:d}) failed for some reason"
        raise Exception(s.format(job_type, job_id))

    # needs to be completed
    elif not ("COMPLETED" in result):
        raise Exception("unknown result in {:s} parsing".format(job_type))

    return


def verify_aces2_completed(path_root, file_path, job_id):
    """verifies that the aces2 output file has completed successfully"""
    final_string = b'The ACES2 program has completed successfully in'
    with open(file_path, "r+b") as source_file:
        with mmap.mmap(source_file.fileno(), 0, prot=mmap.PROT_READ) as memmap_file:
            # the target string should be in the last couple lines of the file
            # first we find the byte location of the begining of the 10th last line
            ten_lines_back = skip_back_n_lines(memmap_file, 10, len(memmap_file) - 1)
            # then we search from there to the end of the file for our target string
            if memmap_file.find(final_string, ten_lines_back) == -1:
                # if we don't find it then the calculation was a failure
                s = (   "Output for ACES2 jobid={:d} did not complete successfully!\n"
                        "Check output file\n{:s}"
                    )
                log.error(s.format(job_id, file_path))

                # verify that the slurm job didn't fail
                job_boss.check_slurm_output(path_root, job_id)

                s = (   "Could not find an issue with slurm output in file\nslurm-{:d}.out\n"
                        "It appears that slurm job didn't fail but ACES2 did?\n"
                        "This appears to be a theory/numerical/symmetry issue.\n"
                    )
                # only the aces2 job failed?
                raise Exception(s.format(job_id))

    return


def submit_job(parameter_dictionary):
    """wrapper for job_boss job submission"""

    job = "sbatch"
    job += (" --mem={memory_size:d}G"
            " --ntasks=1"
            " --job-name={file_name:s}"
            " --cpus-per-task={cpus:d}"
            " --workdir={work_dir:s}"
            # " --output={output:s}/%x.o%j"
            " --partition={partition:s}"
            " --export="
            # "MK_SCRATCH_DIR={make_scratch:s}"
            # ",SCRATCH_COPY_IN={scratch_in:s}"
            # ",SCRATCH_COPY_OUT={scratch_out:s}"
            ",HEADNODE={hostname:s}"
            ",PREAMBLE={pre_amble:s}"
            ",JOB={job_name:s}"
            ",POSTAMBLE={post_amble:s}"
            ",PARENT_PID={parent_pid:d}"
            " /home/ngraymon/chem740/work_dir/submit_template"
        )

    job_id, out, error = job_boss.submit_job(job, parameter_dictionary)

    # check for any errors
    if error.decode('utf-8') is not "":
        s = "Failed to execute script \n{:s}\n due to {:s}"
        raise Exception(s.format(command, error.decode('utf-8')))

    return job_id
# -----------------------------------------------------------


def parse_input_parameters(file_path = None):
    return # no workey
    # Default is to assume parameter file is in the same directory
    if file_path is None:
        file_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(file_path, default_file_in["parameter"].format("default"))

    # where we store the data
    data_dictionary = {
        "calculation"   : None,
        "zmat path"     : None,
        "theory"        : None,
        "basis"         : None,
        }

    # read in the file contents
    try:
        verify_file_exists(file_path)

        with open(file_path, 'r') as source_file:
            for line in source_file:
                if not (line == "" or line.startswith("#")):
                    header = line.strip()
                    data = source_file.readline().strip()
                    if header in ["zmat path", "basis", "theory", "calculation"]:
                        data_dictionary[header] = data
                    else:
                        s = (   "Header {:s} is not valid\n"
                                "Check that your {:s} has the correct formatting"
                            )
                        raise ValueError(s.format(header, file_path))
    except ValueError as error_object:
        raise error_object


    # check the validity of the parameter file
    assert(data_dictionary["basis"] in basis_list)
    assert(data_dictionary["theory"] in theory_list)
    assert(data_dictionary["calculation"] in calculations_list)
    assert(os.path.isfile(data_dictionary["zmat path"]))

    return data_dictionary


def hartree_fock_calculation(path_root, name, zmat, parameter_dictionary, hf_type):
    """creates and populates the input file and submission file for the job"""
    log.flow("Setting up {:s} calculation".format(hf_type))

    memory_size = 10 # in GB's

    # file location
    file_name = default_file_in[hf_type].format(name)
    file_path = os.path.join(path_root, file_name)

    # create the input file template
    template = zmat.get()
    template += "*ACES2( BASIS={basis:s},"
    template += "\n\t" + "CALC=SCF,"
    template += "\n\t" + "MEMORY_SIZE={:d}GB,".format(memory_size)
    # template += "\n\t" + "{DROPMO:s},"
    template += "\n\t" + ")"
    template += "\n"

    # populate the template using the input dictionary
    output = template.format_map(parameter_dictionary)

    with open(file_path, 'w') as dest_file:
        dest_file.write(output)

    # "/scratch/$USER/pimc/data_set_${DATA_SET_ID}/"
    # copy_out = "mv --force /scratch/$USER/pimc/data_set_${DATA_SET_ID}/"
    # path_output = "/scratch/$USER/pimc/data_set_{:d}/execution_output/"

    # nothing for now
    copy_in = ""
    copy_out = ""
    path_output = ""

    slurm_parameters = {
        "memory_size" : memory_size+3,
        "file_name" : file_name,
        "cpus" : 1,
        "work_dir" : path_root,
        "output" : path_output,
        "partition" : job_boss.partition,
        "make_scratch" : 0,
        "scratch_in" : copy_in,
        "scratch_out" : copy_out,
        "pre_amble" : "jobaces",
        "job_name" : file_name.replace(".in", ""),
        "post_amble" : "",
        "hostname" : job_boss.hostname,
        "parent_pid" : os.getpid(),
    }

    # run the job
    job_id = submit_job(slurm_parameters)

    wait_on_results(job_id, "SCF")

    file_name = default_file_out[hf_type].format(name)
    file_path = os.path.join(path_root, file_name)

    verify_aces2_completed(path_root, file_path, job_id)

    return


def parse_hartree_fock_output(vibron, execution_state):
    """extracts relevant data from output of job"""
    hf_type = str(State(execution_state).name)
    file_name = vibron.file_out[hf_type]
    file_path = os.path.join(vibron.path_root, file_name)

    verify_file_exists(file_path)

    # read the output file from a scf calculation
    # and attempt to choose a DROPMO value
    def estimate_dropmo(file_path):

        # access the file using memory map for efficiency
        with open(file_path, "r+b") as source_file:
            with mmap.mmap(source_file.fileno(), 0, prot=mmap.PROT_READ) as memmap_file:

                # find the begining and ending of the important region
                target_begin = 'ORBITAL EIGENVALUES (ALPHA)'
                target_end = '+++++++++++++++++++++++++++++++++++++++++++'
                begin = find_string_in_file(memmap_file, file_path, target_begin)
                end = find_string_in_file(memmap_file, file_path, target_end)

                # go there
                memmap_file.seek(begin)

                # throw away the header lines
                past_headers = skip_forward_n_lines(memmap_file, 4, memmap_file.tell())
                memmap_file.seek(past_headers)

                # read all the relevant data
                data_in_bytes = memmap_file.read(end - memmap_file.tell())
                data_as_string = data_in_bytes.decode(encoding="utf-8")
                lines = data_as_string.strip().splitlines()

        # attempt to guess a DROPMO value
        try:
            # where we store the data
            list_ev = np.array([line.split(maxsplit=5)[3] for line in lines], dtype=float)
            log.debug("list_ev "+str(list_ev))

            # the differences in each pair of eV's
            diffs = list(map(lambda e: (e[0] / e[1]), pairwise(list_ev)))
            log.debug("diffs "+str(diffs))

            # biggest gap in eV's
            max_diff = max(diffs)
            log.debug("max_diff: "+str(max_diff))

            # getguess_dropmo
            guess_dropmo = diffs.index(max_diff)+1
            log.debug("guess_dropmo: "+str(guess_dropmo))

        except Exception as error:
            guess_dropmo = -1
            log.debug("We failed to guess a dropmo")
            pass

        string_orbital = "Here are the orbital energies\n"
        string_orbital += "-"*30 + "\n"
        string_orbital += "".join([str(ev)+"\n" for ev in list_ev])
        string_orbital += "+"*30

        # show the user the orbital energies
        print(string_orbital)

        # ask user if our guess of dropmo is appropriate
        if not guess_dropmo == -1:
            s = "We guessed that dropmo should be {:d} is that acceptable?\n"
            valid = get_yes_no_from_user(s.format(guess_dropmo))
        else:
            print("We failed to guess a dropmo value")
            valid = False

        # if the dropmo value is bad then get a new value from the user
        while not valid:
            guess_dropmo = get_integer_from_user("Please provide an alternative dropmo value\n")
            s = "You provided a new dropmo value of {:d} is that acceptable?\n"
            valid = get_yes_no_from_user(s.format(guess_dropmo))


        # if zero then don't put any typeword
        string_dropmo = ""

        if guess_dropmo == 1:
            string_dropmo = "DROPMO=1"

        if guess_dropmo > 1:
            # a range of orbitals is indicated by '1-X'
            string_dropmo = "DROPMO=1-{:d}".format(guess_dropmo)

        return string_dropmo

    # read the output file from a scf calculation
    # and attempt to choose a NIP value
    def estimate_NIP(file_path):
        # access the file using memory map for efficiency
        with open(file_path, "r+b") as source_file:
            with mmap.mmap(source_file.fileno(), 0, prot=mmap.PROT_READ) as memmap_file:

                # find the begining and ending of the important region
                target_begin = 'ORBITAL EIGENVALUES (ALPHA)'
                target_end = '+++++++++++++++++++++++++++++++++++++++++++'
                begin = find_string_in_file(memmap_file, file_path, target_begin)
                end = find_string_in_file(memmap_file, file_path, target_end)

                # go there
                memmap_file.seek(begin)

                # throw away the header lines
                past_headers = skip_forward_n_lines(memmap_file, 4, memmap_file.tell())
                memmap_file.seek(past_headers)

                # read all the relevant data
                data_in_bytes = memmap_file.read(end - memmap_file.tell())
                data_as_string = data_in_bytes.decode(encoding="utf-8")
                lines = data_as_string.strip().splitlines()

        # grab the guess_dropmo as a number
        string_dropmo = vibron.parameter_dictionary["DROPMO"]
        if string_dropmo == "":
            guess_dropmo = 0
        else:
            guess_dropmo = int(string_dropmo[-1])
        print(guess_dropmo)

        # attempt to guess a NIP value
        try:
            # where we store the data
            list_ev = np.array([lin.split(maxsplit=5)[3] for lin in lines], dtype=float)
            log.debug("list_ev "+str(list_ev))

            # the differences in each pair of eV's
            diffs = list(map(lambda e: (e[0] / e[1]), pairwise(list_ev)))
            log.debug("diffs "+str(diffs))

            if guess_dropmo >= len(list_ev):
                raise Exception("Can't remove orbitals that don't exist")

            list_dropped = np.delete(list_ev, range(0, guess_dropmo+1))
            log.debug("list_dropped "+str(list_dropped))

            # the differences in each pair of eV's
            diffs2 = list(map(lambda e: (e[0] / e[1]), pairwise(list_dropped)))
            log.debug("diffs2 "+str(diffs2))

            # biggest gap in eV's
            second_highest = max(diffs2)
            log.debug("second_highest: "+str(second_highest))

            # the suggested nip value
            guess_nip = len(diffs) - diffs.index(second_highest)
            log.debug("guess_nip: "+str(guess_nip))

        except Exception as error:
            guess_nip = -1
            log.debug("We failed to guess a nip")
            pass


        string_orbital = "Here are the orbital energies\n"
        string_orbital += "-"*30 + "\n"
        string_orbital += "".join([str(ev)+"\n" for ev in list_ev])
        string_orbital += "+"*30

        # show the user the orbital energies
        print(string_orbital)


        # ask user if our guess of nip is appropriate
        if not guess_nip == -1:
            s = "We guessed that nip should be {:d} is that acceptable?\n"
            valid = get_yes_no_from_user(s.format(guess_nip))
        else:
            print("We failed to guess a nip value")
            valid = False


        # if the nip value is bad then get a new value from the user
        while not valid:
            guess_nip = get_integer_from_user("Please provide an alternative nip value\n")
            s = "You provided a new nip value of {:d} is that acceptable?\n"
            valid = get_yes_no_from_user(s.format(guess_nip))


        return guess_nip

    # this could possibly be redesigned
    # first scf estimates the DROPMO parameter
    if hf_type is "DROPMO":
        # save the estimate in the vibron object
        vibron.parameter_dictionary["DROPMO"] = estimate_dropmo(file_path)

    # second scf estimates the new DROPMO parameter
    # and checks which orbitals to include at the optimized geometry
    elif hf_type is "NIP":
        # vibron.parameter_dictionary["DROPMO"] = estimate_dropmo(file_path)
        vibron.parameter_dictionary["nip"] = estimate_NIP(file_path)

    else:
        s = (   "Tried to call parse_hartree_fock_output "
                "with hf_type argument of {:s} which is not implemented"
            )
        raise NotImplementedError(s.format(hf_type))

    return


def geometry_optimization(path_root, name, zmat, parameter_dictionary):
    """creates and populates the input file and submission file for the job"""
    log.flow("Setting up calculation")

    # in GB's
    memory_size = 20

    # file location
    file_name = default_file_in["geometry"].format(name)
    file_path = os.path.join(path_root, file_name)

    # create the input file template
    template = zmat.get(opt_zmat=True)
    template += "*ACES2( BASIS={basis:s},"
    template += "\n\t" + "CALC={theory:s},"
    template += "\n\t" + "MEMORY_SIZE={:d}GB,".format(memory_size)
    # template += "\n\t" + "{DROPMO:s},"
    if zmat.kind == 'cartesian':
        template += "\n\t" + "GEOM_OPT=RIC,"
        # template += "\n\t" + "GEOM_OPT=CART,"
    template += "\n\t" + ")"
    template += "\n"

    # populate the template using the input dictionary
    output = template.format_map(parameter_dictionary)

    # write the input file
    with open(file_path, 'w') as dest_file:
        dest_file.write(output)

    # nothing for now
    # copy_in = ""
    # copy_out = ""
    # path_output = ""

    slurm_parameters = {
        "memory_size" : memory_size+3,
        "file_name" : file_name,
        "cpus" : 1,
        "work_dir" : path_root,
        "output" : "",
        "partition" : job_boss.partition,
        "make_scratch" : "",
        "scratch_in" : "",
        "scratch_out" : "",
        "pre_amble" : "jobaces",
        "job_name" : file_name.replace(".in", ""),
        "post_amble" : "",
        "hostname" : job_boss.hostname,
        "parent_pid" : os.getpid(),
    }

    # run the job
    job_id = submit_job(slurm_parameters)

    wait_on_results(job_id, "geometry optimization")

    file_name = default_file_out["geometry"].format(name)
    file_path = os.path.join(path_root, file_name)

    verify_aces2_completed(path_root, file_path, job_id)
    return


def parse_opt_output(vibron):
    """extract substrings describing the optimized geometry from the output file generated by a geometry optimization calcultion"""
    file_name = vibron.file_out["geometry"]
    file_path = os.path.join(vibron.path_root, file_name)

    verify_file_exists(file_path)

    def extract_internal(file_path):
        """assume that the substrings describing the geometry are in internal coordinates"""
        with open(file_path, "r+b") as source_file:
            # access the file using memory map for efficiency
            with mmap.mmap(source_file.fileno(), 0, prot=mmap.PROT_READ) as memmap_file:

                # find the begining of the important region
                target_string = 'Summary of optimized internal coordinates'
                begin = find_string_in_file(memmap_file, file_path, target_string)

                # go there
                memmap_file.seek(begin)

                # throw away the headers
                memmap_file.readline()
                memmap_file.readline()

                lines = []
                while True:
                    line = memmap_file.readline().decode(encoding="utf-8")
                    if line in ["\n", "Frequencies of the updated Hessian at convergence"]:
                        break

                    lines.append(line)

                new_internal_geometry = dict([map(str.strip, line.split('=')) for line in lines])

        return new_internal_geometry


    def extract_cartesian(file_path):
        """assume that the substrings describing the geometry are in cartesian coordinates"""
        with open(file_path, "r+b") as source_file:
            # access the file using memory map for efficiency
            with mmap.mmap(source_file.fileno(), 0, prot=mmap.PROT_READ) as memmap_file:

                # find the begining of the important region
                target_string = 'Summary of optimized Cartesian coordinates (Ang)'
                begin = find_string_in_file(memmap_file, file_path, target_string)

                # go there
                memmap_file.seek(begin)

                # throw away the headers
                memmap_file.readline()
                memmap_file.readline()

                lines = []
                while True:
                    line = memmap_file.readline().decode(encoding="utf-8")
                    if line in ["\n", "Frequencies of the updated Hessian at convergence"]:
                        break

                    lines.append(line)

                # create list of updated geometry
                list_of_lines = [x.split() for x in lines]

                new_cartesian_geometry = ""

                for line in list_of_lines:
                    for item in line:
                        new_cartesian_geometry += str(item) + " "
                    new_cartesian_geometry += "\n"
                new_cartesian_geometry += "\n"

        return new_cartesian_geometry


    # return optimized geometry
    if vibron.zmat.kind is "internal":
        vibron.zmat.set(extract_internal(file_path))
    else:
        vibron.zmat.set(extract_cartesian(file_path))
    return


def vibrational_frequency(path_root, name, zmat, parameter_dictionary):
    """creates and populates the input file and submission file for the job"""
    log.flow("Setting up calculation")

    # in GB's
    memory_size = 20

    # file location
    file_name = default_file_in["vib"].format(name)
    file_path = os.path.join(path_root, file_name)

    # create the input file template
    template = zmat.get()
    template += "*ACES2( BASIS={basis:s},"
    template += "\n\t" + "CALC={theory:s},"
    template += "\n\t" + "MEMORY_SIZE={:d}GB,".format(memory_size)
    # template += "\n\t" + "{DROPMO:s},"
    template += "\n\t" + "VIB=FINDIF,"
    template += "\n\t" + ")"
    template += "\n"

    # populate the template using the input dictionary
    output = template.format_map(parameter_dictionary)

    with open(file_path, 'w') as dest_file:
        dest_file.write(output)

    # nothing for now
    copy_in = ""
    copy_out = ""
    path_output = ""

    slurm_parameters = {
        "memory_size" : memory_size+3,
        "file_name" : file_name,
        "cpus" : 1,
        "work_dir" : path_root,
        "output" : "",
        "partition" : job_boss.partition,
        "make_scratch" : "",
        "scratch_in" : "",
        "scratch_out" : "",
        "pre_amble" : "jobaces",
        "job_name" : file_name.replace(".in", ""),
        "post_amble" : "fcm",
        "hostname" : job_boss.hostname,
        "parent_pid" : os.getpid(),
    }

    # run the job
    job_id = submit_job(slurm_parameters)

    wait_on_results(job_id, "vibrational frequency")

    file_name = default_file_out["vib"].format(name)
    file_path = os.path.join(path_root, file_name)

    verify_aces2_completed(path_root, file_path, job_id)

    # make sure the output files we expect to be present are present
    # verify_file_exists(os.path.join(path_root, "fcmint"))
    # verify_file_exists(os.path.join(path_root, file_name))
    # verify_file_exists(os.path.join(path_root, file_name))
    # verify_file_exists(os.path.join(path_root, file_name))
    # verify_file_exists(os.path.join(path_root, file_name))
    return


def ip_calculation(path_root, name, zmat, parameter_dictionary):
    """creates and populates the input file and submission file for the job"""
    log.flow("Setting up calculation")

    # in GB's
    memory_size = 10

    # file location
    file_name = default_file_in["ipeomcc"].format(name)
    file_name = file_name.format(parameter_dictionary["nip"])
    file_path = os.path.join(path_root, file_name)

    template = zmat.get()
    template += "*ACES2( BASIS={basis:s},"
    template += "\n\t" + "CALC={theory:s},"
    template += "\n\t" + "MEMORY_SIZE={:d}GB,".format(memory_size)
    # template += "\n\t" + "{DROPMO:s},"
    template += "\n\t" + "IP_CALC=IP_EOMCC,"
    template += "\n\t" + ")"
    template += "\n"
    template += "\n" + "*mrcc_gen"
    template += "\n\t" + "nip={nip:d}"
    #template += "\n\t" + "ip_low=-25"
    #template += "\n\t" + "ea_high=0"
    template += "\n" + "*end"
    template += "\n"

    # populate the template using the input dictionary
    output = template.format_map(parameter_dictionary)

    with open(file_path, 'w') as dest_file:
        dest_file.write(output)

    # nothing for now
    copy_in = ""
    copy_out = ""
    path_output = ""

    slurm_parameters = {
        "memory_size" : memory_size+3,
        "file_name" : file_name,
        "cpus" : 1,
        "work_dir" : path_root,
        "output" : "",
        "partition" : job_boss.partition,
        "make_scratch" : "",
        "scratch_in" : "",
        "scratch_out" : "",
        "pre_amble" : "jobaces",
        "job_name" : file_name.replace(".in", ""),
        "post_amble" : "",
        "hostname" : job_boss.hostname,
        "parent_pid" : os.getpid(),
    }

    # run the job
    job_id = submit_job(slurm_parameters)

    wait_on_results(job_id, "IP_EOM")

    file_name = default_file_out["ipeomcc"].format(name)
    file_name = file_name.format(parameter_dictionary["nip"])
    file_path = os.path.join(path_root, file_name)

    verify_aces2_completed(path_root, file_path, job_id)
    return


def parse_ip_output(vibron):
    """extracts relevant data from output of job"""
    file_name = vibron.file_out["ipeomcc"].format(vibron.parameter_dictionary["nip"])
    file_path = os.path.join(vibron.path_root, file_name)

    verify_file_exists(file_path)


    def verify_percent_singles(file_path, cutoff):
        """read the output file from an ionization potential calculation extract the \% singles and verify they they are above a cutoff"""
        with open(file_path, "r+b") as source_file:
            # access the file using memory map for efficiency
            with mmap.mmap(source_file.fileno(), 0, prot=mmap.PROT_READ) as memmap_file:

                # find the begining and ending of the important region
                target_begin = 'Summary of ionization-potential eom-cc calculation'
                begin = find_string_in_file(memmap_file, file_path, target_begin)

                # go there
                memmap_file.seek(begin)

                # throw away the header lines
                past_headers = skip_forward_n_lines(memmap_file, 5, memmap_file.tell())
                memmap_file.seek(past_headers)

                # store the lines with the relevant information
                lines = []
                while True:
                    line = memmap_file.readline().decode(encoding="utf-8")
                    if( line is "\n" or "--------" in line):
                        break

                    lines.append(line)

                # extract the relevant information
                percent_singles = np.array([x.split(maxsplit=5)[4] for x in lines], dtype=float)
                boolean_array = (percent_singles <= vibron.singles_cutoff).copy()

                # verify that all %singles are above our cutoff point
                if not boolean_array.any():
                    return True

                # otherwise record a warning
                s = "We found {:d} states with less than {:f}\% singles"
                log.warning(s.format(np.count_nonzero(boolean_array), vibron.singles_cutoff))
                return False

    return verify_percent_singles(file_path, vibron.singles_cutoff)


def verify_ip_states(job):
    '''run calculations to verify the range of states selected'''
    while True:
        job.ip_calc()

        if parse_ip_output(job):
            s = "We verified that {:d} states had \% singles above {:f}"
            log.info(s.format(job.parameter_dictionary["nip"], job.singles_cutoff))
            if job.state is State.IPEOMCC:
                job.advance() # this is a problem
            # we only want to advance if we arn't already at prepVibron or vibron calc
            break

        s = "Reducing the nip from {:d} to {:d}"
        log.debug(s.format(job.parameter_dictionary["nip"], job.parameter_dictionary["nip"]-1))

        job.parameter_dictionary["nip"] -= 1
        if job.parameter_dictionary["nip"] == 0:
            s = "We failed to get % singles above 85% and ended up with a nip of 0"
            raise Exception(s) # Don't know how to handle this right now
    return


def prepVibron_calculation(path_root, name, zmat, parameter_dictionary):
    """creates and populates the input file and submission file for the job"""
    log.flow("Setting up calculation")

    # in GB's
    memory_size = 10

    # file location
    file_name = default_file_in["prepVib"].format(name)
    file_path = os.path.join(path_root, file_name)

    # create the input file template
    template = zmat.get()
    template += "*ACES2( BASIS={basis:s},"
    template += "\n\t" + "CALC={theory:s},"
    template += "\n\t" + "MEMORY_SIZE={:d}GB,".format(memory_size)
    # template += "\n\t" + "{DROPMO:s},"
    template += "\n\t" + "IP_CALC=IP_EOMCC,"
    template += "\n\t" + "PREP_VIBRON=ON,"
    template += "\n\t" + "GRID_VIBRON={gridVibron:d},"
    template += "\n\t" + "CC_CONV={ccConv:d},"
    template += "\n\t" + "SCF_CONV={scfConv:d},"
    template += "\n\t" + "ESTATE_TOL={estateTol:d},"
    template += "\n\t" + ")"
    template += "\n"
    template += "\n" + "*mrcc_gen"
    template += "\n\t" + "nip={nip:d}"
    template += "\n" + "*ip_eom"
    template += "\n\t" + "diabatize=on"
    template += "\n\t" + "heff_low=0"
    template += "\n\t" + "heff_high=200"
    template += "\n" + "*end"
    template += "\n"

    # populate the template using the input dictionary
    output = template.format_map(parameter_dictionary)

    with open(file_path, 'w') as dest_file:
        dest_file.write(output)

    # nothing for now
    copy_in = ""
    copy_out = ""
    path_output = ""

    slurm_parameters = {
        "memory_size" : memory_size+3,
        "file_name" : file_name,
        "cpus" : 2,
        "work_dir" : path_root,
        "output" : "",
        "partition" : job_boss.partition,
        "make_scratch" : "",
        "scratch_in" : "",
        "scratch_out" : "",
        "pre_amble" : "jobaces",
        "job_name" : file_name.replace(".in", ""),
        "post_amble" : "vibron",
        "hostname" : job_boss.hostname,
        "parent_pid" : os.getpid(),
    }

    # run the job
    job_id = submit_job(slurm_parameters)

    wait_on_results(job_id, "preparing for vibron")

    job_boss.check_slurm_output(path_root, job_id)

    verify_file_exists(os.path.join(path_root, "fcmfinal"))
    verify_file_exists(os.path.join(path_root, default_file_out["pntheff"].format(name)))
    verify_file_exists(os.path.join(path_root, default_file_out["vibronCp"].format(name)))
    verify_file_exists(os.path.join(path_root, default_file_out["vibronIn"].format(name)))
    return


def parse_prepVibron_output(vibron):
    """extracts relevant data from output of job"""
    def remove_extra_heff(file_name, target_string='HEFF_IP2'):
        """remove non relevant data from output files kind argument does nothing at the moment"""
        file_path = os.path.join(vibron.path_root, file_name)
        # access the file using memory map for efficiency
        with open(file_path, "r+b") as source_file:
            with mmap.mmap(source_file.fileno(), 0) as memmap_file:
                # store the original file size
                size_original = memmap_file.size()
                log.debug(size_original)
                log.debug(target_string)

                # find the first occurance of the target_string
                start_in_bytes = find_string_in_file(memmap_file, file_path, target_string)
                log.debug(start_in_bytes)
                # get the byte location of the start of the line
                start_in_bytes = memmap_file.rfind(b'\n', 0, start_in_bytes)
                # we want to ignore that newline character however
                start_in_bytes += 1
                log.debug(start_in_bytes)

                # get the EOF location in numBytes
                memmap_file.seek(0, os.SEEK_END)
                end_of_file_in_bytes = memmap_file.tell()
                log.debug(end_of_file_in_bytes)

                # the length of our source
                length_in_bytes = end_of_file_in_bytes - start_in_bytes
                log.debug(length_in_bytes)

                # copy the file, resize it and write the change to disk
                memmap_file.move(0, start_in_bytes, length_in_bytes)
                memmap_file.resize(length_in_bytes)
                memmap_file.flush()

                # # store the original file size
                # size_original = memmap_file.size()

                # # find the begining and ending of the important region
                # destStart = find_string_in_file(memmap_file, file_path, target_string)
                # destEnd = rfind_string_in_file(memmap_file, file_path, target_string)
                # log.debug(destStart, destEnd)

                # # find the byte location of the end of the line
                # destStart = memmap_file.rfind(b'\n', 0, destStart)
                # destEnd = memmap_file.find(b'\n', destEnd)
                # log.debug(destStart, destEnd)

                # # if we can't find a newline character
                # # that means the substring was found
                # # in the first line of the file
                # if destStart == -1:
                #     destStart = 0

                # destEnd += 1
                # log.debug(destStart, destEnd)

                # # find the start of the section to copy
                # start_in_bytes = memmap_file.find(b'HEFF_IP2', destEnd)
                # log.debug(start_in_bytes)

                # # get the location of the begining of the line
                # start_in_bytes = memmap_file.rfind(b'\n', 0, start_in_bytes)
                # log.debug(start_in_bytes)

                # # get the EOF location in numBytes
                # memmap_file.seek(0, os.SEEK_END)
                # EOF = memmap_file.tell()

                # # check file sizes
                # destLength = destEnd - destStart
                # length_in_bytes = EOF - start_in_bytes

                # log.debug(memmap_file.size(), EOF)
                # log.debug(destStart, start_in_bytes, length_in_bytes)
                # log.debug(size_original - destLength)

                # memmap_file.move(destStart, start_in_bytes, length_in_bytes)
                # memmap_file.resize(size_original - destLength)
                # memmap_file.flush()
        return


    def add_nip_to_file(file_name, nip):
        """modify file to contain the nip"""
        file_path = os.path.join(vibron.path_root, file_name)
        with open(file_path, "r+b") as destFile:
            # access the file using memory map for efficiency
            with mmap.mmap(destFile.fileno(), 0) as memmap_file:

                # we assume that there is only one 'xxx' substring in the whole file
                startTarget = 'xxx'
                start = find_string_in_file(memmap_file, file_path, startTarget)
                # go there
                memmap_file.seek(start)
                s = "{:>6,d}"
                # attempt to write 3 bytes over the 'xxx'
                try:
                    bytesWritten = memmap_file.write(s.format(nip).encode(('utf-8')))
                    assert(bytesWritten == 6) # make sure we wrote the right amount
                    memmap_file.flush() # write the changes from memory to disk
                except (Exception, ValueError) as e:
                    s = "Failed to write the nip={:d} to {:s}"
                    log.error(s.format(nip, file_path))
                    raise e    # 1
        return


    def copy_output_file(source, destination):
        """ verifies source file exists before copying it to destination  """
        path_source = os.path.join(vibron.path_root, source)
        verify_file_exists(path_source)
        path_destination = os.path.join(vibron.path_root, destination)
        shutil.copyfile(path_source, path_destination)
        return

    file_new = "PNTHEFF"
    copy_output_file(vibron.file_out["pntheff"], file_new)
    remove_extra_heff(file_new)
    # remove_extra_heff(file_new, 'HEFF_0')

    file_new = vibron.file_in["vibronCp"]
    copy_output_file(vibron.file_out["vibronCp"], file_new)
    remove_extra_heff(file_new)
    # remove_extra_heff(file_new, 'Heff_0')

    file_new = vibron.file_in["vibronIn"]
    copy_output_file(vibron.file_out["vibronIn"], file_new)
    add_nip_to_file(file_new, vibron.parameter_dictionary["nip"])

    return


def vibron_calculation(path_root, name, zmat, parameter_dictionary):
    """creates and populates the input file and submission file for the job"""
    log.flow("Setting up calculation")

    # in GB's
    memory_size = 15

    # file location
    file_name = default_file_in["vibronIn"].format(name)

    # nothing for now
    copy_in = ""
    copy_out = ""
    path_output = ""

    slurm_parameters = {
        "memory_size" : memory_size+3,
        "file_name" : file_name,
        "cpus" : 1,
        "work_dir" : path_root,
        "output" : "",
        "partition" : job_boss.partition,
        "make_scratch" : "",
        "scratch_in" : "",
        "scratch_out" : "",
        "pre_amble" : "jobvibron",
        "job_name" : file_name.replace(".in", ""),
        "post_amble" : "",
        "hostname" : job_boss.hostname,
        "parent_pid" : os.getpid(),
    }

    # run the job
    job_id = submit_job(slurm_parameters)

    wait_on_results(job_id, "VIBRON")

    job_boss.check_slurm_output(path_root, job_id)
    return


def parse_vibron_output(vibron):
    """extracts relevant data from output of job"""
    file_name = vibron.file_out["vibron"]
    file_path = os.path.join(vibron.path_root, file_name)

    verify_file_exists(file_path)

    final_string = b'All done in main_vibron'
    with open(file_path, "r+b") as source_file:
        with mmap.mmap(source_file.fileno(), 0, prot=mmap.PROT_READ) as memmap_file:
            # the target string should be in the last couple lines of the file
            # first we find the byte location of the begining of the 10th last line
            ten_lines_back = skip_back_n_lines(memmap_file, 10, len(memmap_file) - 1)
            # then we search from there to the end of the file for our target string
            if memmap_file.find(final_string, ten_lines_back) == -1:
                # if we don't find it then the calculation was a failure
                s = "The vibron calculation failed to successfully complete"\
                  + "\nCheck output file {:s}"
                # log.error(s.format(file_path))
                raise Exception(s.format(file_path))
            else:
                log.info("vibron calculation successfully completed!")

    return


class ZmatClass:
    """handles all details related to zmat file which represents the geometry of the molecule can hold internal or cartesian coordinates"""

    kind = ["cartesian", "internal"][0]

    # string that holds the relational ZMAT info
    # in internal coordinates (without the bond/angle values)
    _internal = None

    # dictionary that holds the coordinate bond/angle values
    # for internal coordinates type of ZMAT
    _coord = None

    # string that holds the relational ZMAT info
    # in cartesian coordinates
    _cartesian = None


    def __init__(self, file_path):
        verify_file_exists(file_path)

        # parse file
        with open(file_path, 'r') as source_file:
            # make sure there is at least one newline at the end
            data = source_file.read() + "\n"

            # determine which coordinates are used in the ZMAT
            if "=" in data:
                self.kind = "internal"

            # process the input
            if self.kind is "internal":
                # split along the blank line
                b = data.split("\n\n")

                # this is the relational ZMAT info
                self._internal = b[0] + "\n"

                # create a dictionary that maps the angles/bonds to their values
                self._coord = dict([map(str.strip, x.split('=')) for x in b[1].splitlines()])

            elif self.kind is "cartesian":
                self._cartesian = data + "\n"

            else:
                s = "Invalid ZMAT file located at {:s}"
                raise Exception(s.format(file_path))

        return


    def get(self, opt_zmat=False):
        """returns a string which represents the coordinates"""

        # creates a string from zmat and self.coord
        # that represents a valid ZMAT file
        if self.kind is "internal":
            completeZMAT = self._internal + "\n"

            # if we arn't optimizing then remove the asteriks
            if not opt_zmat:
                completeZMAT = completeZMAT.replace("*", "")

            # add the geometry values
            for key, value in self._coord.items():
                completeZMAT += key+"="+value+"\n"

            # need a blank line coordinate list
            completeZMAT += "\n"
            return completeZMAT

        # no modification is needed for the cartesian version
        elif self.kind is "cartesian":
            return self._cartesian


    def set(self, geometry_new):
        """updates geometry data"""

        # just update the dict values
        if self.kind is "internal":
            self._coord.update(geometry_new)

        # special treatment is necessary
        elif self.kind is "cartesian":
            geometry_new = [line for line in geometry_new.split("\n") if line != '']
            geometry_old = [line for line in self._cartesian.split("\n") if line != '']
            diff = len(geometry_old) - len(geometry_new)

            if diff == 0:
                self._cartesian = "\n".join(geometry_new) + "\n\n"

            elif diff > 0:
                for i in range(0, len(geometry_new)):
                    geometry_old[i+diff] = geometry_new[i]

                self._cartesian = "\n".join(geometry_old) + "\n\n"

            else:
                s = "The new geometries have MORE? atoms than before???"\
                  + "Old geometry\n{:s}\nNew Geometry\n{:s}\n"
                raise Exception(s.format(self._cartesian, geometry_new))


            return

        return


class VibronExecutionClass:
    """handles all details related to the execution"""

    # the molecule of interest
    name = None

    # ZmatClass object that contains the ZMAT information
    zmat = None

    # the directory in which we execute jobs
    path_root = None

    # parameters
    parameter_dictionary = {
        "calculation"   : None,
        "theory"        : None,
        "basis"         : None,
        "DROPMO"        : "",
        "NIP"           : None,
        "gridVibron"    : 3, # or 4
        "ccConv"        : 10,
        "scfConv"       : 10,
        "estateTol"     : 10,
        }

    file_in = {}
    file_out = {}

    # our cut off for % singles
    singles_cutoff = 85.00

    # our cut off for % triples
    triples_cutoff = 92.00

    # State enum that corresponds to location in execution workflow
    state = None

    def __init__(self, name, root, state=None):
        self.name = name
        self.path_root = root
        self.file_in = default_file_in.copy()
        self.file_out = default_file_out.copy()
        self.path_state = self.path_root + file_name_state


        # fill in names
        for key, val in self.file_in.items():
            self.file_in[key] = val.format(self.name)
        for key, val in self.file_out.items():
            self.file_out[key] = val.format(self.name)

        if state is not None:
            self.state = State(int(state))
        return


    def record_state(self):
        with open(self.path_state, 'a') as file_state:
            string_state = str(self.state.name)+"\n"
            file_state.write(string_state)
        return


    def advance(self):
        """increase state counter and log it IF not already at the highest state"""
        # create state file if it doesn't exist
        if not os.path.isfile(self.path_state):
            s = "State file {:s} doesn't exist, creating a blank file"
            log.debug(s.format(self.path_state))
            open(self.path_state, 'w').close()

        self.record_state()

        # advance or write down that we are complete
        if self.state == State.FINISHED:
            # we have finished!
            self.record_state()
        elif self.state.value < State.max():
            old_state = self.state
            self.state = State(old_state.value + 1)
            s = "Advancing from state {:s} to state {:s}"
            log.debug(s.format(old_state.name, self.state.name))
        else:
            raise Exception("This code should not be executed")

        return


    def setup_zmat(self, file_path=None):
        """assumes that you formatted the ZMAT file correctly"""
        # Default is to assume parameter file is in /path_root/*
        if file_path is None:
            file_path = os.path.join(   self.path_root,
                                        # "parameters",
                                        self.file_in["zmat"]
                                    )

        self.zmat = ZmatClass(file_path)
        return


    def parse_input_parameters(self, file_path = None):
        """loads data into the object"""
        # Default is to assume parameter file is in /path_root/*
        if file_path is None:
            file_path = os.path.join(   self.path_root,
                                        # "parameters",
                                        self.file_in["parameter"]
                                    )

        # attempt to read in file
        try:
            verify_file_exists(file_path)

            with open(file_path, 'r') as source_file:
                for line in source_file:
                    # if line is not blank AND is not a comment
                    if bool(line.strip()) ^ bool(line.startswith("#")):
                        header = line.strip()
                        data = next(source_file).strip()

                        if header in self.parameter_dictionary.keys():
                             self.parameter_dictionary[header] = data

                        else:
                            s = (   "Header {:s} is not valid\n"
                                    "Check that your {:s} has the correct formatting"
                                    )
                            raise ValueError(s.format(header, file_path))

        except ValueError as error_object:
            raise error_object

        # check the validity of the parameter file
        assert(self.parameter_dictionary["basis"] in basis_list)
        assert(self.parameter_dictionary["theory"] in theory_list)
        assert(self.parameter_dictionary["calculation"] in calculations_list)

        # create state file if it doesn't exist
        if not os.path.isfile(self.path_state):
            s = "State file {:s} doesn't exist, creating a blank file"
            log.debug(s.format(self.path_state))
            open(self.path_state, 'w').close()

        # check the state of the last calculation
        if self.state is None:
            log.debug("Attempting to set state using state file")

            if os.stat(self.path_state).st_size == 0:
                s = "State file {:s} is empty, defaulting to initial State"
                log.debug(s.format(self.path_state))
                self.state = State(State.min())
                return

            with open(self.path_state, 'r') as file_state:
                lines = file_state.readlines()
                for line in reversed(lines):

                    if line in ["", "\n"]:
                        continue

                    old_state = str(line.strip())

                    if old_state == State(State.max()):
                        self.state = State(State.max())

                    elif old_state in State:
                        self.state = State(State(old_state).value + 1)

                    for member in State:
                        if member.name == str(line.strip()):
                            self.state = State(member.value + 1)
                            return
                    else:
                        s = (   "State file {:s} contained an invalid state: {:s}"
                                "Defaulting to initial state"
                            )
                        log.debug(s.format(line, self.path_state))
                        self.state = State(State.min())

                    return
        return


    def hf(self, matching_state):
        """wrapper method"""
        if self.state == matching_state:
            hartree_fock_calculation(self.path_root, self.name, self.zmat,
                                        self.parameter_dictionary, self.state.name)
            self.advance()
        return

    def geom(self):
        """wrapper method"""
        if self.state == State.OPT:
            geometry_optimization(self.path_root, self.name, self.zmat, self.parameter_dictionary)
            self.advance()
        return

    def vib(self):
        """wrapper method"""
        if self.state == State.VIB:
            vibrational_frequency(self.path_root, self.name, self.zmat, self.parameter_dictionary)
            self.advance()
        return

    def ip_calc(self):
        """wrapper method"""
        if self.state == State.IPEOMCC:
            ip_calculation(self.path_root, self.name, self.zmat, self.parameter_dictionary)
        return

    def prepVibron(self):
        """wrapper method"""
        if self.state == State.PREPVIB:
            prepVibron_calculation(self.path_root, self.name, self.zmat, self.parameter_dictionary)
            self.advance()
        return

    def vibron_calc(self):
        """wrapper method"""
        if self.state == State.VIBRON:
            vibron_calculation(self.path_root, self.name, self.zmat, self.parameter_dictionary)
            self.advance()
        return

    def calculate_vibronic_model(self):
        # prepare signal handler - this is used to synchronize with the submitted jobs
        signal.signal(signal.SIGUSR1, job_boss.SIGUSR1_handle)

        self.parse_input_parameters()
        self.setup_zmat()

        if self.state.value < State.max():
            # self.hf(State.DROPMO)
            # parse_hartree_fock_output(self, State.DROPMO)

            self.geom()
            parse_opt_output(self)

            self.hf(State.NIP)
            parse_hartree_fock_output(self, State.NIP)

            self.vib()

            verify_ip_states(self)

            self.prepVibron()
            parse_prepVibron_output(self)

            self.vibron_calc()
            parse_vibron_output(self)
            self.advance() # important to do after parsing vibron_output
            log.info("We have reached the end of execution.")
        return


def test_one(molecule_name = "ch2o", inital_job_state=0):
    """runs the job in the current directory stores results in folder named by the molecule"""

    # create execution directory
    path_root = os.path.dirname(os.path.realpath(__file__))
    path_root = os.path.join(path_root, molecule_name)
    os.makedirs(path_root,  exist_ok=True)

    job = VibronExecutionClass(molecule_name, path_root, inital_job_state)
    job.calculate_vibronic_model()
    return


# need a better naming - distinguishing scheme between
# build_vibronic_model and calculate_vibronic_model
def calculate_vibronic_model_wrapper_one(data_set_num, inital_job_state=0):
    """executes jobs using exists data_set_# file structure"""

    # create execution directory
    path_root = '/work/ngraymon/pimc/data_set_{:d}/electronic_structure/'
    os.makedirs(path_root,  exist_ok=True)

    job = VibronExecutionClass(molecule_name, path_root, inital_job_state)
    job.calculate_vibronic_model()
    return

if (__name__ == "__main__"):

    # we provide a specific named file
    if len(sys.argv) is 2:
        test_one(str(sys.argv[1]))

    # we provide a specific named file
    # and we provide a integer paramter
    # which indicates what stage of the calculation to start at
    elif len(sys.argv) is 3:
        test_one(str(sys.argv[1]), int(sys.argv[2]))

    else:
        test_one()
