"""provides a number of job submission related functions - a helper module for job submission
assumes the server uses SLURM"""

# system imports
import subprocess
import threading
import socket
import signal
import time
import sys
import os

# third party imports

# local imports
# import pibronic.data.vibronic_model_io as vIO
# from ..constants import GB_per_byte, maximum_memory_per_node
from ..log_conf import log
from .. import constants
from .. import pimc
# from ..server.server import ServerExecutionParameters as SEP

# lock for asynchronous communication
job_state_lock = threading.Lock()
job_almost_done_flag = False

# this should be redone
# partition = 'highmem' if hostname == 'feynman' else 'serial'


def get_path_to_python_executable():
    """returns the absolute path to the python executable currently executing this script"""
    return os.path.abspath(os.__file__)


def get_path_of_job_boss_directory():
    """returns the absolute path to the directory holding job_boss.py """
    return os.path.abspath(__file__)


def subprocess_submit_asynch_wrapper(cmd, **kwargs):
    """ wrapper for subprocess.Popen function to allow for different implementation for different python versions"""
    if sys.version_info[:2] >= (3, 7):
        return subprocess.Popen(cmd, text=True, **kwargs)
    if (3, 5) <= sys.version_info[:2] <= (3, 7):
        return subprocess.Popen(cmd, universal_newlines=True, **kwargs)


def subprocess_run_wrapper(cmd, **kwargs):
    """ wrapper for subprocess.run function to allow for different implementation for different python versions"""
    if sys.version_info[:2] >= (3, 7):
        return subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if (3, 5) <= sys.version_info[:2] <= (3, 7):
        return subprocess.run(cmd, universal_newlines=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              **kwargs)


def get_hostname():
    """returns the hostname of the cluster of the server (from SLURM) as a string"""
    cmd = ['scontrol', 'show', 'config']
    result = subprocess_run_wrapper(cmd)

    for line in result.stdout.splitlines():
        if "ClusterName" in line:
            return line.split('=')[1].strip()
    else:
        raise Exception("Did not find ClusterName in the config data from scontrol!?")


def SIGUSR1_handle(signum, frame):
    """signal handler - uses job_state_lock to record signal using the bool job_almost_done_flag"""
    global job_almost_done_flag
    log.lock("I got the signal")

    # attempt to acquire the lock
    if job_state_lock.acquire(blocking=False):
        log.lock("I was able to acquire the lock")
        job_almost_done_flag = (signum == signal.SIGUSR1)
        job_state_lock.release()
    # if we can't acquire the lock then set an alarm
    else:
        log.lock("I couldn't aquire the lock so I'm setting an alarm")
        job_almost_done_flag = (signum == signal.SIGUSR1)
        signal.alarm(10)  # set the alarm to wake up this thread
    return


def check_acct_state(id_job):
    """returns the recorded state of the job (from SLURM) as a string"""
    cmd = ['sacct', '-n', '-o', 'state', '-j', str(id_job)]
    result = subprocess_run_wrapper(cmd)
    return result.stdout


def check_running_state(id_job):
    """returns the running state of the job (from SLURM) as a string"""
    cmd = ['scontrol', 'show', 'job', str(id_job)]
    result = subprocess_run_wrapper(cmd)
    return result.stdout, result.stderr


def synchronize_with_job(id_job, job_type="default"):
    """synchronizes with a submitted job """
    log.lock(f"Synchronizing with job (id={id_job:})")

    # if the job is in the queue wait for the signal
    out, error_state = check_running_state(id_job)
    if error_state == '':
        log.lock("About to Enter critical section")
        with job_state_lock:
            if not job_almost_done_flag:
                # wait for the signal or the alarm
                log.lock(f"Waiting on \'{job_type:s}\' job (id={id_job:})")
                signal.sigwait([signal.SIGUSR1, signal.SIGALRM])
                log.lock("Woke up")

            log.lock("About to leave the critical section")
    else:
        raise Exception(f"Undefined behaviour, the job state was:\n {error_state:s}")

    # Wait until the job's state file reflects a successful execution, or an error with execution
    while True:
        out, error_state = check_running_state(id_job)

        # if this string is in the job's state file then it successfully executed
        if "COMPLETED" in out:
            break

        # If the "COMPLETED" string is not in the job's state file AND scontrol reports no errors
        # the most likely cause is the state file has not been updated since the job left the queue
        # therefore we should wait until the state file is updated
        elif error_state == '':
            time.sleep(5)

        # if the error_state from scontrol is not empty then some undefined behaviour occurred
        else:
            raise Exception(f"Undefined behaviour, the job state was:\n {error_state:s}")
    return


def check_slurm_output(path_root, id_job):
    """checks ouput file from slurm for errors memory issues, incorrect arguments, etc"""

    # sanity check - sleep so that we are sure slurm has finished writing the output file
    time.sleep(10)

    slurm_path = os.path.join(path_root, f"slurm-{id_job:d}.out")
    log.debug(f"Checking slurm output:\n{slurm_path:}\n")

    with open(slurm_path, "r") as source_file:
        if not ("Finished" in source_file):
            log.debug("We got here too early?")  # should this be an exception?

        elif "SIGSEGV" in source_file:
            raise MemoryError(f"Job {id_job:d} had a Segmentation fault, see file {slurm_path:s}")

        elif "Killed" in source_file:
            # most likely cause is that it ran out of memory
            if "Exceeded step memory limit" in source_file:
                raise MemoryError(f"Job {id_job:d} ran out of memory, see file {slurm_path:s}")
            raise Exception(f"Job {id_job:d} failed for an unknown reason, see file {slurm_path:s}")
        else:
            log.warning(f"Undefined execution, check file {slurm_path:s} for issues")
    return


def extract_id_job_from_output(out):
    """ returns the job id inside the str argument 'out' or None if the job id is not present
    if no job id can be found then a warning is raised"""

    id_job = None
    if isinstance(out, str) and len(out) >= 21:  # this is hardcoded - possibly should be changed
        id_job = int(out[20:])
    else:
        log.warning(f"Not sure how to extract job id from\n{out}\n")

    return id_job


def submit_job(command, parameter_dictionary):
    """craft the job submission command - no error checking"""

    # we should add error checking to the parameter_dictionary here
    command = command.format(**parameter_dictionary)

    """ submits the job to the slurm server"""
    result = subprocess_run_wrapper(command, shell=True)
    result.check_returncode()

    id_job = extract_id_job_from_output(result.stdout)

    return id_job, result.stdout, result.stderr


def assert_partition_exists(partition_name):
    """only checks that the given string is listed as a partition by sinfo"""
    cmd = ['sinfo', '-O', 'partition']
    result = subprocess_run_wrapper(cmd)
    assert partition_name is not None, "The partition string is None?"
    assert partition_name in result.stdout, f"Partition {partition_name} is not present in {result.stdout}"
    return


def serialize_BoxData_dictionary(parameter_dictionary):
    """ wrapper for the call to BoxData's json_serialize()
    takes a dictionary of parameters and returns a string which is a 'serialized' version of those parameters
    when the submitted job eventually executes it initializes a BoxData (or child) object using that 'serialized' string"""
    return pimc.BoxData.json_serialize(params=parameter_dictionary)


def prepare_job_feynman(param_dict):
    """wrapper for job_boss job submission"""
    log.debug("feynman job")

    # for safety
    param_dict["hostname"] = "feynman"

    template_name = ("D{id_data:d}_"
                     "R{id_rho:d}_"
                     "P{number_of_beads:d}_"
                     "T{temperature:.2f}"
                     )

    param_dict["job_name"] = template_name.format(**param_dict)

    template_from = f"\"{param_dict['path_scratch']:}results/{param_dict['job_name']:}_J\""
    template_to = f"\"{param_dict['path_rho']:}results/\""

    param_dict["copy_from"] = template_from
    param_dict["copy_to"] = template_to

    param_dict["execution_parameters"] = serialize_BoxData_dictionary(param_dict)

    job_boss_directory = get_path_of_job_boss_directory()

    export_options = (""
                      " --export="
                      "ROOT_DIR={path_rho:s}"
                      ",SCRATCH_DIR={path_scratch:s}"
                      ",COPY_FROM={copy_from:s}"
                      ",COPY_TO={copy_to:s}"
                      f",PYTHON3_PATH={get_path_to_python_executable()}"
                      f",SAMPLING_SCRIPT={job_boss_directory:}" + "/nlogn_feynman/{script_name}"
                      ",EXECUTION_PARAMETERS=\'{execution_parameters:s}\'"
                      )

    param_dict["export_options"] = export_options.format(**param_dict)

    sbatch = "sbatch"
    sbatch += (
                " -m n"  # this stops all mail from being sent
                # " --priority 0"  # this defines the priority of the job, default is 0
                " --ntasks=1"
                " --job-name={job_name:s}"
                " --partition={partition:s}"
                " -D {path_rho:}execution_output/"
                " --output={path_rho:}execution_output/{job_name:s}.o%A"
                # " --ntask={number_of_tasks:d}"
                " --cpus-per-task={cpus_per_task:d}"
                " --cores-per-socket={cores_per_socket:d}"
                " --mem={memory_per_node:}G"
                # " --mem-per-cpu={memory_per_cpu:}G" # mutually exclusive with --mem
                " {wait_param:s}"  # optional wait parameter
                " {export_options:s}"
                f" {job_boss_directory}/nlogn_feynman/pimc_job.sh"
                )

    return sbatch


def prepare_job_nlogn(param_dict):
    """wrapper for job_boss job submission"""
    log.debug("nlogn job")

    # for safety
    param_dict["hostname"] = "nlogn"

    template_name = ("D{id_data:d}_"
                     "R{id_rho:d}_"
                     "P{number_of_beads:d}_"
                     "T{temperature:.2f}"
                     )

    param_dict["job_name"] = template_name.format(**param_dict)

    template_from = f"\"{param_dict['path_scratch']:}results/\""
    template_to = f"\"{param_dict['path_rho']:}results/\""

    param_dict["copy_from"] = template_from
    param_dict["copy_to"] = template_to

    param_dict["execution_parameters"] = serialize_BoxData_dictionary(param_dict)

    job_boss_directory = get_path_of_job_boss_directory()

    export_options = (""
                      " --export="
                      "ROOT_DIR={path_rho:s}"
                      ",SCRATCH_DIR={path_scratch:s}"
                      ",COPY_FROM={copy_from:s}"
                      ",COPY_TO={copy_to:s}"
                      f",PYTHON3_PATH={get_path_to_python_executable()}"
                      f",SAMPLING_SCRIPT={job_boss_directory:}" + "/nlogn_feynman/{script_name}"
                      ",EXECUTION_PARAMETERS=\'{execution_parameters:s}\'"
                      )

    param_dict["export_options"] = export_options.format(**param_dict)

    sbatch = "sbatch"
    sbatch += (
               " -m n"  # this stops all mail from being sent
               # " --priority 0"  # this defines the priority of the job, default is 0
               " --ntasks=1"
               # " --ntask={number_of_tasks:d}"
               " --job-name={job_name:s}"
               " --partition={partition:s}"
               " -D {path_rho:}execution_output/"
               " --output={path_rho:}execution_output/{job_name:s}.o%A"
               " --cpus-per-task={cpus_per_task:d}"
               " --cores-per-socket={cores_per_socket:d}"
               " --mem={memory_per_node:}G"
               # " --mem-per-cpu={memory_per_cpu:}G" # mutually exclusive with --mem
               " {wait_param:s}"  # optional wait parameter
               " {export_options:s}"
               f" {job_boss_directory}/nlogn_feynman/pimc_job.sh"
               )

    return sbatch


def prepare_job_compute_canada(param_dict):
    """ wrapper for jobs on compute canada servers """
    template_name = ("D{id_data:d}_"
                     "R{id_rho:d}_"
                     "P{number_of_beads:d}_"
                     "T{temperature:.2f}"
                     )

    param_dict["job_name"] = template_name.format(**param_dict)

    template_from = f"\"{param_dict['path_scratch']:}results/{param_dict['job_name']:}_J\""
    template_to = f"\"{param_dict['path_rho']:}results/\""

    param_dict["copy_from"] = template_from
    param_dict["copy_to"] = template_to

    param_dict["execution_parameters"] = serialize_BoxData_dictionary(param_dict)

    job_boss_directory = get_path_of_job_boss_directory()

    export_options = (""
                      " --export="
                      "ROOT_DIR={path_rho:s}"
                      ",SCRATCH_DIR={path_scratch:s}"
                      ",COPY_FROM={copy_from:s}"
                      ",COPY_TO={copy_to:s}"
                      f",PYTHON3_PATH={get_path_to_python_executable()}"
                      f",SAMPLING_SCRIPT={job_boss_directory:}" + "/compute_canada/{script_name}"
                      ",EXECUTION_PARAMETERS=\'{execution_parameters:s}\'"
                      )

    param_dict["export_options"] = export_options.format(**param_dict)

    sbatch = "sbatch"
    sbatch += (
                " -m n"  # this stops all mail from being sent
                # " --priority 0"  # this defines the priority of the job, default is 0
                " --ntasks=1"
                # " --ntask={number_of_tasks:d}"
                " --job-name={job_name:s}"
                # " --partition={partition:s}"  #  don't use partition!!
                " -D {path_rho:}execution_output/"
                " --output={path_rho:}execution_output/{job_name:s}.o%A"
                " --cpus-per-task={cpus_per_task:d}"
                " --cores-per-socket={cores_per_socket:d}"
                " --mem={memory_per_node:}G"
                " --account=rrg-pnroy"
                # " --mem-per-cpu={memory_per_cpu:}G" # mutually exclusive with --mem
                " {wait_param:s}"  # optional wait parameter
                " {export_options:s}"
                f" {job_boss_directory}/compute_canada/pimc_job.sh"
                )

    return sbatch


def prepare_job_orca(param_dict):
    """wrapper for job_boss job submission"""
    log.debug("orca job")

    # for safety
    param_dict["hostname"] = "orca"
    return prepare_job_compute_canada(param_dict)


def prepare_job_graham(param_dict):
    """wrapper for job_boss job submission"""
    log.debug("graham job")

    # for safety
    param_dict["hostname"] = "graham"
    return prepare_job_compute_canada(param_dict)


class SubmissionClass():

    # the largest number of samples to be drawn for an individual job submitted to the server
    # any job that requires more samples is split into multiple job submissions
    MAX_SAMPLES_PER_JOB = int(1E4)

    # default values
    param_dict = {
        "delta_beta": constants.delta_beta,
        "temperature_list": [0.0, ],
        "bead_list": [0, ],
        "number_of_samples": 0,
        "number_of_samples_overall": 0,
        "number_of_samples_per_job": 0,
        "number_of_blocks": 0,
        "number_of_states": 0,
        "number_of_modes": 0,
        "number_of_beads": 0,
        "number_of_links": 0,
        "path_scratch": "",
        "path_root": "",
        "path_data": "",
        "path_rho": "",
        "id_data": 0,
        "id_rho": 0,
        "partition": None,
        "hostname": "",
        "block_size": 1,
        "memory_per_node": 1,
        "total_memory": 1,
        "cpus_per_task": 1,
        "cores_per_socket": 1,
        "wait_param": "",
        "script_name": None,
    }

    node_dict = {
        "feynman": prepare_job_nlogn,
        "nlogn": prepare_job_nlogn,
        "graham": prepare_job_graham,
        "orca": prepare_job_orca,
    }

    def prepare_paths(self):
        """ fill the param_dict with values from the give FileStructure """

        # this function may not be needed in the future?
        new = {
                "path_scratch": self.FS.path_rho.replace("work", "scratch"),
                "path_root": self.FS.path_root,
                "path_data": self.FS.path_data,
                "path_rho": self.FS.path_rho,
                "path_vib_model": self.FS.path_vib_model,
                "path_rho_model": self.FS.path_rho_model,
        }
        self.param_dict.update(new)
        return

    def __init__(self, input_FS, input_param_dict=None):
        """ takes a FileStructure object and an optional parameter dictionary - no error checking at the moment """

        # set the default hostname when initialized
        self.param_dict['hostname'] = get_hostname()

        if input_param_dict is not None:
            self.param_dict.update(input_param_dict)

        self.FS = input_FS
        self.prepare_paths()
        return

    def verify_hostname_is_valid(self, hostname):
        """ this checks the hostname against a pre-defined dictionary
        this should alert the user if they are trying to submit jobs on a server
        without first preparing a job submission wrapper
        """
        if hostname in self.node_dict:
            return True

        raise Exception(f"Hostname {hostname} is undefined - please confirm that a prepare_job_$HOSTNAME function has been defined in job_boss.py and that the respective hostname is present in the dictionary node_dict defined in job_boss.py\n")

    def construct_job_command(self, params):
        """ this """
        assert self.verify_hostname_is_valid(params["hostname"]), "You shouldn't see this!"
        return self.node_dict[params["hostname"]](params)


class PimcSubmissionClass(SubmissionClass):
    """ Class to store all the logic involved with submitting 1 or more jobs to a server running SLURM - should be self consistent """

    # the largest number of samples to be drawn for an individual job submitted to the server
    # any job that requires more samples is split into multiple job submissions
    MAX_SAMPLES_PER_JOB = int(1E5)

    # TODO - this method could most likely be improved upon
    # default values
    param_dict = {
        # "delta_beta": constants.delta_beta,
        "temperature_list": [0.0, ],
        "bead_list": [0, ],
        "number_of_samples": 0,
        "number_of_samples_overall": 0,
        "number_of_samples_per_job": 0,
        "number_of_blocks": 0,
        "number_of_states": 0,
        "number_of_modes": 0,
        "number_of_beads": 0,
        "path_scratch": "",
        "path_root": "",
        "path_data": "",
        "path_rho": "",
        "id_data": 0,
        "id_rho": 0,
        "partition": None,
        "hostname": None,
        "block_size": int(1e3),
        "memory_per_node": 20,
        "total_memory": 20,
        "cpus_per_task": 4,
        "cores_per_socket": 4,
        "wait_param": "",
        "script_name": "pimc.py",
    }

    def prepare_paths(self):
        """ fill the param_dict with values from the give FileStructure """
        super().prepare_paths()

    def __init__(self, input_FS, input_param_dict=None):
        """ takes a FileStructure object and an optional parameter dictionary - no error checking at the moment """
        super().__init__(input_FS, input_param_dict)

    def setup_blocks_and_jobs(self):
        """ calculates the following:
        - the number of blocks per job
        - the number of samples per job
        - the number of jobs
        and stores them in the param_dict
        """
        n_samples = self.param_dict["number_of_samples"]
        block_size = self.param_dict["block_size"]
        assert isinstance(n_samples, int) and isinstance(block_size, int)

        assert block_size <= n_samples, f"block size {block_size} must be less than or equal to the number of samples {n_samples}"
        assert block_size <= self.MAX_SAMPLES_PER_JOB, f"block size {block_size} must be less than or equal to the maximum number of samples per job {self.MAX_SAMPLES_PER_JOB}"

        # calculate how many samples we need for each job
        samples_per_job = min(n_samples, self.MAX_SAMPLES_PER_JOB)
        self.param_dict["number_of_samples_per_job"] = samples_per_job
        self.param_dict["number_of_samples"] = samples_per_job

        # calculate how many blocks we need for each job
        blocks_per_job = samples_per_job // block_size
        self.param_dict["number_of_blocks"] = blocks_per_job

        # calculate how many jobs we need
        total_samples = self.param_dict["number_of_samples_overall"]

        # TODO - HACKY
        if total_samples is 0:
            total_samples = n_samples
            self.param_dict["number_of_samples_overall"] = n_samples

        self.n_jobs = max(1, total_samples // self.MAX_SAMPLES_PER_JOB)
        return

    def submit_jobs(self):
        """submit jobs at diff temps and beads"""

        self.setup_blocks_and_jobs()

        temperature_list = self.param_dict["temperature_list"]
        bead_list = self.param_dict["bead_list"]

        import copy
        for T in temperature_list:
            for P in bead_list:

                params = copy.deepcopy(self.param_dict)
                params["number_of_beads"] = P
                params["number_of_links"] = P
                params["temperature"] = T

                # clean up this area
                hostname = get_hostname()
                params["hostname"] = hostname

                if hostname == "feynman" or hostname == "nlogn":
                    if params["partition"] is None:  # only c
                        params["partition"] = 'highmem' if hostname == 'feynman' else 'serial'

                if hostname == "orca" or hostname == "graham":
                    params["partition"] = None

                if params["partition"] is not None:
                    log.flow(f'Hostname {hostname}\nPartition {params["partition"]}\n')
                    assert_partition_exists(params["partition"])

                command = self.construct_job_command(params)

                for sample_index in range(0, self.n_jobs):
                    print(T, P, sample_index)
                    module_name = self.__class__.__module__
                    job_id, out, error = sys.modules[module_name].submit_job(command, params)
                    print(f"Job ID:{job_id:}")
        return

    def submit_job_array(self):
        """for each temp submit an array of jobs over the beads"""
        assert False, "this is currently under development"

        temperature_list = self.param_dict["temperature_list"]
        # bead_list = self.param_dict["bead_list"]

        for temp in temperature_list:
            log.info("Submitting jobarray")

            # TODO -  what were base_params supposed to be?
            # params = base_params.copy()
            # params["temperature"] = temp

            # if job_boss.hostname == "nlogn":
            #     job_id = submit_job_nlogn(params)
            # elif job_boss.hostname == "feynman":
            #     job_id = submit_job_feynman(params)

            # print(job_boss.hostname)
        return
