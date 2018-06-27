"""provides a number of job submission related functions - a helper module for job submission
assumes the server uses SLURM"""

# system imports
import subprocess
from subprocess import Popen
# from subprocess import TimeoutExpired
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

# this should go in the param_dict
hostname = socket.gethostname()
# this should be redone
partition = 'highmem' if hostname == 'feynman' else 'serial'


def SIGUSR1_handle(signum, frame):
    """signal handler - uses job_state_lock to record signal using the bool job_almost_done_flag"""
    global job_almost_done_flag
    log.lock("I got the signal")

    # attempt to aquire the lock
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
    cmd = ['/usr/bin/sacct', '-n', '-o', 'state', '-j', str(id_job)]
    capture_object = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return capture_object.stdout.decode('utf-8')


def check_running_state(id_job):
    """returns the running state of the job (from SLURM) as a string"""
    cmd = ['/usr/bin/scontrol', 'show', 'job', str(id_job)]
    capture_object = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return capture_object.stdout.decode('utf-8'), capture_object.stderr.decode('utf-8')


def synchronize_with_job(id_job, job_type="default"):
    """synchronizes with a submitted job """
    log.lock("Synchronizing with job (id={:})".format(id_job))

    # if the job is in the queue wait for the signal
    out, error_state = check_running_state(id_job)
    if error_state == '':
        log.lock("About to Enter critical section")
        with job_state_lock:
            if not job_almost_done_flag:
                # wait for the signal or the alarm
                log.lock("Waiting on \'{:s}\' job (id={:})".format(job_type, id_job))
                signal.sigwait([signal.SIGUSR1, signal.SIGALRM])
                log.lock("Woke up")

            log.lock("About to leave the critical section")
    else:
        s = "Undefined behaviour, the job state was:\n {:s}"
        raise Exception(s.format(error_state))

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

        # if the error_state from scontrol is not empty then some undefined behaviour occured
        else:
            s = "Undefined behaviour, the job state was:\n {:s}"
            raise Exception(s.format(error_state))
    return


def check_slurm_output(path_root, id_job):
    """checks ouput file from slurm for errors memory issues, incorrect arguments, etc"""

    # sanity check - sleep so that we are sure slurm has finished writing the output file
    time.sleep(10)

    slurm_path = os.path.join(path_root, "slurm-{:d}.out".format(id_job))
    log.debug("Checking slurm " + slurm_path)

    with open(slurm_path, "r") as source_file:
        if not ("Finished" in source_file):
            log.debug("We got here too early?")  # should this be an exception?

        elif "SIGSEGV" in source_file:
            s = "Job {:d} had a Segmentation fault, see file {:s}"
            raise MemoryError(s.format(id_job, slurm_path))

        elif "Killed" in source_file:
            # most likely cause is that it ran out of memory
            if "Exceeded step memory limit" in source_file:
                s = "Job {:d} ran out of memory, see file {:s}"
                raise MemoryError(s.format(id_job, slurm_path))
            s = "Job {:d} failed for an unknown reason, see file {:s}"
            raise Exception(s.format(id_job, slurm_path))

        else:
            s = "Undefined execution, check file {:s} for issues"
            log.warning(s.format(slurm_path))
    return


def submit_job(command, parameter_dictionary):
    """craft the job submission command - no error checking"""

    # we should add error checking to the parameter_dictionary here
    command = command.format(**parameter_dictionary)

    """ submits the job to the slurm server"""
    p = Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, error = p.communicate()

    # extract the id_job
    id_job = None
    if len(out.decode()) >= 21:  # this is hardcoded - possibly should be changed
        id_job = int(out.decode()[20:])

    return id_job, out, error


def assert_partition_exists(partition_name):
    """only checks that the given string is listed as a partition by sinfo"""
    cmd = ['/usr/bin/sinfo', '-O', 'partition']
    capture_object = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = capture_object.stdout.decode('utf-8')
    assert partition_name is not None, "The partition string is None?"
    assert partition_name in out, f"Partition {partition_name} is not present in {out}"
    return


class PimcSubmissionClass:
    """ Class to store all the logic involved with submitting 1 or more jobs to a server running SLURM - should be self consistent """

    # the largest number of samples to be drawn for an individual job submitted to the server
    # any job that requires more samples is split into multiple job submissions
    MAX_SAMPLES_PER_JOB = int(1E5)

    # TODO - this method could most likely be improved upon
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
        "path_scratch": "",
        "path_root": "",
        "path_data": "",
        "path_rho": "",
        "id_data": 0,
        "id_rho": 0,
        "partition": "serial",
        "hostname": None,
        # "partition": None,
        # "partition": "serial,highmem",
        "block_size": int(1e2),
        "memory_per_node": 20,
        "total_memory": 20,
        "cpus_per_task": 4,
        "cores_per_socket": 4,
        "wait_param": "",
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

        if input_param_dict is not None:
            self.param_dict.update(input_param_dict)

        self.FS = input_FS
        self.prepare_paths()
        return

    def construct_job_command(self, params):
        """ x """
        node_dict = {
            "feynman": prepare_job_nlogn,
            "nlogn": prepare_job_nlogn,
        }

        if params["hostname"] in node_dict:
            return node_dict[params["hostname"]](params)
        else:
            raise Exception(f"Hostname {hostname} is undefined - please check job_boss to see\n")

    def setup_blocks_and_jobs(self):
        """ calculates the following:
        - the number of blocks per job
        - the number of samples per job
        - the number of jobs
        and stores them in the param_dict
        """
        n_samples = self.param_dict["number_of_samples"]
        block_size = self.param_dict["block_size"]
        assert type(n_samples) is int and type(block_size) is int

        assert block_size < n_samples, f"block size {block_size} must be less than the number of samples {n_samples}"
        assert block_size < self.MAX_SAMPLES_PER_JOB, f"block size {block_size} must be less than the maximum number of samples per job {self.MAX_SAMPLES_PER_JOB}"

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
                params["temperature"] = T

                params["hostname"] = socket.gethostname()
                if params["partition"] is None:
                    params["partition"] = 'highmem' if hostname == 'feynman' else 'serial'

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


def prepare_job_feynman(param_dict):
    """wrapper for job_boss job submission"""
    log.debug("feynman job")

    # for saftey
    param_dict["hostname"] = "feynman"

    template_name = ("D{id_data:d}_"
                     "R{id_rho:d}_"
                     # "A{number_of_states:d}_"
                     # "N{number_of_modes:d}_"
                     # "X{number_of_samples:d}_"
                     "P{number_of_beads:d}_"
                     "T{temperature:.2f}"
                     )

    param_dict["job_name"] = template_name.format(**param_dict)

    template_from = "\"{:}results/{:}_J\""
    template_from = template_from.format(param_dict["path_scratch"],
                                         param_dict["job_name"],
                                         # param_dict["path_rho"],
                                         )
    template_to = "\"{:}results/\"".format(param_dict["path_rho"])

    param_dict["copy_from"] = template_from
    param_dict["copy_to"] = template_to
    # TODO - remove circular dependancy?
    from ..pimc import BoxData
    param_dict["execution_parameters"] = BoxData.json_serialize(params=param_dict)

    export_options = (""
                      " --export="
                      "ROOT_DIR={path_rho:s}"
                      ",SCRATCH_DIR={path_scratch:s}"
                      ",COPY_FROM={copy_from:s}"
                      ",COPY_TO={copy_to:s}"
                      ",PYTHON3_PATH=/home/ngraymon/.dev/ubuntu_18.04/bin/python3"
                      ",SAMPLING_SCRIPT=/home/ngraymon/pibronic/pibronic/server/pimc.py"
                      ",EXECUTION_PARAMETERS=\'{execution_parameters:s}\'"
                      )

    param_dict["export_options"] = export_options.format(**param_dict)

    # print(param_dict["hostname"])

    sbatch = "sbatch"
    sbatch += (
                " -m n"  # this stops all mail from being sent
                " --priority 0"  # this defines the priority of the job, default is 0
                " --ntasks=1"
                " --job-name={job_name:s}"
                " --partition={partition:s}"
                " --workdir={hostname:s}:{path_rho:}execution_output/"
                " --output={path_rho:}execution_output/{job_name:s}.o%A"
                # " --ntask={number_of_tasks:d}"
                " --cpus-per-task={cpus_per_task:d}"
                " --cores-per-socket={cores_per_socket:d}"
                " --mem={memory_per_node:}G"
                # " --mem-per-cpu={memory_per_cpu:}G" # mutually exclusive with --mem
                " {wait_param:s}"  # optional wait parameter
                " {export_options:s}"
                " /home/ngraymon/pibronic/pibronic/server/pimc_job.sh"
                )

    return sbatch


def prepare_job_nlogn(param_dict):
    """wrapper for job_boss job submission"""
    log.debug("nlogn job")

    # for saftey
    param_dict["hostname"] = "nlogn"

    template_name = ("D{id_data:d}_"
                     "R{id_rho:d}_"
                     # "A{number_of_states:d}_"
                     # "N{number_of_modes:d}_"
                     # "X{number_of_samples:d}_"
                     "P{number_of_beads:d}_"
                     "T{temperature:.2f}"
                     )

    param_dict["job_name"] = template_name.format(**param_dict)

    # template_from = "\"{:}results/"
    template_from = "\"{:}results/\""
    template_from = template_from.format(param_dict["path_scratch"],
                                         # param_dict["job_name"],
                                         # param_dict["path_rho"],
                                         )
    template_to = "\"{:}results/\"".format(param_dict["path_rho"])

    param_dict["copy_from"] = template_from
    param_dict["copy_to"] = template_to

    # del param_dict["delta_beta"]
    # del param_dict["number_of_samples_per_job"]
    # del param_dict["number_of_samples_overall"]
    param_dict["execution_parameters"] = pimc.BoxData.json_serialize(params=param_dict)

    export_options = (""
                      " --export="
                      "ROOT_DIR={path_rho:s}"
                      ",SCRATCH_DIR={path_scratch:s}"
                      ",COPY_FROM={copy_from:s}"
                      ",COPY_TO={copy_to:s}"
                      ",PYTHON3_PATH=/home/ngraymon/.dev/ubuntu_18.04/bin/python3"
                      # ",PYTHON3_PATH=/home/ngraymon/test/local/bin/python3"
                      ",SAMPLING_SCRIPT=/home/ngraymon/test/Pibronic/pibronic/server/pimc.py"
                      ",EXECUTION_PARAMETERS=\'{execution_parameters:s}\'"
                      )

    param_dict["export_options"] = export_options.format(**param_dict)

    sbatch = "sbatch"
    sbatch += (
               " -m n"  # this stops all mail from being sent
               " --priority 0"  # this defines the priority of the job, default is 0
               " --ntasks=1"
               " --job-name={job_name:s}"
               " --partition={partition:s}"
               " --workdir={hostname:s}:{path_rho:}execution_output/"
               " --output={path_rho:}execution_output/{job_name:s}.o%A"
               # " --ntask={number_of_tasks:d}"
               " --cpus-per-task={cpus_per_task:d}"
               " --cores-per-socket={cores_per_socket:d}"
               " --mem={memory_per_node:}G"
               # " --mem-per-cpu={memory_per_cpu:}G" # mutually exclusive with --mem
               " {wait_param:s}"  # optional wait parameter
               " {export_options:s}"
               " /home/ngraymon/test/Pibronic/pibronic/server/pimc_job.sh"
               )

    return sbatch

#
#
#
#
#
