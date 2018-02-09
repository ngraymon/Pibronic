# job_boss.py
#
# system imports
import subprocess
from subprocess import Popen, TimeoutExpired
import threading
import socket
import signal
import time
import os

# third party imports

# local imports
# from ..data import vibronic_model_io as vIO
from ..log_conf import log
from .. import constants

# lock for asynchronous communication
job_state_lock = threading.Lock()
job_almost_done_flag = False

# which headnode we are on
hostname = socket.gethostname()
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
        signal.alarm(10) # set the alarm to wake up this thread
    return


def check_acct_state(id_job):
    """get the recorded state of the job"""
    cmd = ['/usr/bin/sacct', '-n', '-o', 'state', '-j', str(id_job)]
    capture_object = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return capture_object.stdout.decode('utf-8')


def check_running_state(id_job):
    """get the running state of the job"""
    cmd = ['/usr/bin/scontrol', 'show', 'job', str(id_job)]
    capture_object = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return capture_object.stdout.decode('utf-8'), capture_object.stderr.decode('utf-8')


def synchronize_with_job(id_job, job_type="default"):
    """sychronizes with a submitted job"""
    log.lock("Synchronizing with job (id={:})".format(id_job))

    # if the job is in the queue wait for the signal
    out, error_state = check_running_state(id_job)
    if error_state == '':
        log.lock("entering critical section")
        with job_state_lock:
            if not job_almost_done_flag:
                # wait for the signal or the alarm
                log.lock("Waiting on \'{:s}\' job (id={:})".format(job_type, id_job))
                signal.sigwait([signal.SIGUSR1, signal.SIGALRM])
                log.lock("woke up")

            log.lock("leaving the critical section")
    else:
        s = "Undefined behaviour,the job state was:\n {:s}"
        raise Exception(s.format(error_state))

    # confirm job has left the queue

    while True:
        out, error_state = check_running_state(id_job)

        if "COMPLETED" in out:
            break

        elif error_state == '':
            time.sleep(5)

        else:
            s = "Undefined behaviour, the job state was:\n {:s}"
            raise Exception(s.format(error_state))
    return


def check_slurm_output(path_root, id_job):
    """checks ouput file from slurm  for errors memory issues, incorrect arguments, etc"""
    # sleep so that slurm finishes writing?
    time.sleep(10)

    slurm_path = os.path.join(path_root, "slurm-{:d}.out".format(id_job))
    log.debug("Checking slurm " + slurm_path)

    with open(slurm_path, "r") as source_file:
        if not ("Finished" in source_file):
            log.debug("We got here too early?")

        elif "SIGSEGV" in source_file:
            s = "Job {:d} had a Segmentation fault, see file {:s}"
            raise MemoryError(s.format(id_job, slurm_path))

        elif "Killed" in source_file:
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
    command = command.format(**parameter_dictionary)

    """ submits the job to the slurm server"""
    p = Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, error = p.communicate()

    # extract the id_job
    id_job = None
    if len(out.decode()) >= 21:
        id_job = int(out.decode()[20:])

    return id_job, out, error


class PimcSubmissionClass:
    """x"""
    LARGEST_JOB_SAMPLE = int(1E5)
    # default values
    param_dict = {
        "delta_beta" : constants.delta_beta,
        "temperature_list" : [0.0,],
        "bead_list" : [0, ],
        "number_of_samples" : 0,
        "number_of_blocks" : 0,
        "number_of_states" : 0,
        "number_of_modes" : 0,
        "number_of_beads" : 0,
        "path_scratch" : "",
        "path_root" : "",
        "path_data" : "",
        "path_rho" : "",
        "id_data" : 0,
        "id_rho" : 0,
        "partition" : "serial,highmem",
        "block_size" : int(1e2),
        "memory_per_node" : 20,
        "total_memory" : 20,
        "cpus_per_task" : 4,
        "cores_per_socket" : 4,
        "wait_param" : "",
    }


    def __init__(self, file_structure_obj, param_dict):
        """x"""
        self.file_struct = file_structure_obj
        # self.path_root = file_structure_obj.path_root
        # self.path_data = file_structure_obj.path_data
        # self.path_rho = self.path_data + "rho_{id_rho:d}/".format(**param_dict)
        self.param_dict.update(param_dict)

        # blocks = self.param_dict["number_of_samples"] // self.param_dict["block_size"]
        blocks = self.LARGEST_JOB_SAMPLE // self.param_dict["block_size"]

        new = {
                "number_of_blocks" : blocks,
                "path_scratch" : self.file_struct.path_rho.replace("work", "scratch"),
                "path_root" : self.file_struct.path_root,
                "path_data" : self.file_struct.path_data,
                "path_rho" : self.file_struct.path_rho,
                "path_model_vib" : self.file_struct.path_vib_params + "coupled_model.json",
                "path_model_rho" : self.file_struct.path_rho_params + "sampling_model.json",
        }

        self.param_dict.update(new)
        return


    def submit_jobs(self):
        """submit jobs at diff temps and beads"""
        temperature_list = self.param_dict["temperature_list"]
        bead_list = self.param_dict["bead_list"]

        # each job only takes x samples
        total_samples = self.param_dict["number_of_samples"]
        self.param_dict["number_of_samples"] = self.LARGEST_JOB_SAMPLE

        num = total_samples // self.LARGEST_JOB_SAMPLE

        for temp in temperature_list:
            for beads in bead_list:

                params = self.param_dict.copy()
                params["number_of_beads"] = beads
                params["temperature"] = temp

                command = None
                if job_boss.hostname == "nlogn":
                    command = prepare_job_nlogn(params)
                elif job_boss.hostname == "feynman":
                    command = prepare_job_feynman(params)
                else:
                    raise Exception("Not on nlogn or feynman?")

                for sample_index in range(0, num):
                    print(temp, beads, sample_index)
                    job_id, out, error = job_boss.submit_job(command, params)
                    # sys.exit(0)
        return


    def submit_job_array(self):
        """for each temp submit an array of jobs over the beads"""

        temperature_list = self.param_dict["temperature_list"]
        bead_list = self.param_dict["bead_list"]

        for temp in temperature_list:
            log.info("Submitting jobarray")

            params = base_params.copy()
            params["temperature"] = temp

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

    template_name = (   "D{id_data:d}_"
                        "R{id_rho:d}_"
                        # "A{number_of_states:d}_"
                        # "N{number_of_modes:d}_"
                        # "X{number_of_samples:d}_"
                        "P{number_of_beads:d}_"
                        "T{temperature:.2f}"
                        )



    param_dict["job_name"] = template_name.format(**param_dict)

    template_from = "\"{:}results/{:}_J\""
    template_from = template_from.format(   param_dict["path_scratch"],
                                            param_dict["job_name"],
                                            # param_dict["path_rho"],
                                        )
    template_to = "\"{:}results/\"".format(param_dict["path_rho"])

    param_dict["copy_from"] = template_from
    param_dict["copy_to"] = template_to
    param_dict["execution_parameters"] = BoxData.json_encode(params=param_dict)

    export_options = (  ""
                        " --export="
                        "ROOT_DIR={path_rho:s}"
                        ",SCRATCH_DIR={path_scratch:s}"
                        ",COPY_FROM={copy_from:s}"
                        ",COPY_TO={copy_to:s}"
                        ",EXECUTION_PARAMETERS=\'{execution_parameters:s}\'"
                        ",PYTHON3_PATH=/home/ngraymon/.dev/ubuntu/16.04/bin/python3"
                        ",SAMPLING_SCRIPT=/home/ngraymon/pibronic/pibronic/server/pimc.py"
                        )

    param_dict["export_options"] = export_options.format(**param_dict)

    print(param_dict["hostname"])

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
                " {wait_param:s}" # optional wait parameter
                " {export_options:s}"
                " /home/ngraymon/pibronic/pibronic/server/pimc_job.sh"
                )

    return sbatch


def prepare_job_nlogn(param_dict):
    """wrapper for job_boss job submission"""
    log.debug("nlogn job")

    # for saftey
    param_dict["hostname"] = "nlogn"

    template_name = (   "D{id_data:d}_"
                        "R{id_rho:d}_"
                        # "A{number_of_states:d}_"
                        # "N{number_of_modes:d}_"
                        # "X{number_of_samples:d}_"
                        "P{number_of_beads:d}_"
                        "T{temperature:.2f}"
                        )

    param_dict["job_name"] = template_name.format(**param_dict)

    template_copy = "\"mv --force \"{:}results/{:}\"* \"{:}results/\" \""
    template_copy = template_copy.format(   param_dict["path_scratch"],
                                            param_dict["job_name"],
                                            param_dict["path_rho"],
                                        )

    param_dict["copy_commands"] = template_copy
    param_dict["execution_parameters"] = BoxData.json_encode(params=param_dict)


    export_options = (  ""
                        " --export="
                        "ROOT_DIR={path_rho:s}"
                        ",SCRATCH_DIR={path_scratch:s}"
                        ",COPY_COMMANDS={copy_commands:s}"
                        ",EXECUTION_PARAMETERS=\'{execution_parameters:s}\'"
                        ",PYTHON3_PATH=/home/ngraymon/.dev/ubuntu/16.04/bin/python3"
                        ",SAMPLING_SCRIPT=/home/ngraymon/pibronic/pibronic/server/pimc.py"
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
                " {wait_param:s}" # optional wait parameter
                " {export_options:s}"
                " /home/ngraymon/pibronic/pibronic/server/pimc_job.sh"
                )

    return sbatch
