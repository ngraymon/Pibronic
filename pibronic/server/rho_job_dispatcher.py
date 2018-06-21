""" submission script for sos jobs (using the sampling model - rho) on a server"""

# system imports
import subprocess
import socket
import os

# third party imports

# local imports
from ..vibronic import vIO
from ..data import file_structure
from ..constants import GB_per_byte, maximum_memory_per_node

# array_cmd = "sbatch"
# array_cmd += (""
#                " --output={hostname:}:{root_dir:}execution_output/"  # defines output directions
#                " --job-name=\"RHO{id_data:d}_B{basis_size:d}_S{n_surfaces:d}_N{n_modes:d}\""  # name the jobs
#                " --array=0-{max_jobs:d}%1"  # this creates a job array with MAX_JOBS number of jobs
#                " --cpu_per_task=16"
#                " -N1"
#                # " --ntasks=1"
#                # " --label"
#                " --mem-per-cpu=1000"
#                # " -n=5"
#                " ./server_scripts/slurm/pimc_job.sh"
#                )

sos_cmd = "sbatch"
sos_cmd += (" -m n"  # this stops all mail from being sent
            " --priority 0"  # this defines the priority of the job, default is 0
            " --mem={memory:}G"
            # " --mem-per-cpu={memory:}G"
            " --ntasks=1"
            " --cpus-per-task={n_cpus:d}"
            # " --workdir={hostname:}:{dir_data:}execution_output/"  # defines output directions
            " --workdir={dir_data:}execution_output/"  # defines output directions
            " --job-name=\"SOS{id_data:d}_B{basis_size:d}_S{n_surfaces:d}_N{n_modes:d}\""
            " --output=\"{dir_data:}execution_output/SOS{id_data:d}_B{basis_size:d}_S{n_surfaces:d}_N{n_modes:d}.o%A\""
            " --export="
            "\"MODES={coupled_modes:d}\""
            ",\"SURFACES={coupled_surfaces:d}\""
            ",\"BASIS_SIZE={basis_size:d}\""
            ",\"NUM_OF_TEMPS={n_temps:d}\""
            ",\"ROOT_DIR={dir_data:}\""
            ",\"NUMBER_OF_CORES={n_cpus:d}\""
            ",\"MEMORY_RESERVED={memory:}\""
            ",\"id_data={id_data:}\""
            ",\"BLAS_MODULE_DIR=/home/ngraymon/dev/privatemodules/openBLAS\""
            ",\"Q_HOSTNAME={hostname:}\""
            " ./server_scripts/sos_job.sh"
            )


rho_cmd = "sbatch"
rho_cmd += (" -m n"  # this stops all mail from being sent
            " --priority 0"  # this defines the priority of the job, default is 0
            " --mem={memory:}G"
            # " --mem-per-cpu={memory:}G"
            " --ntasks=1"
            " --cpus-per-task={n_cpus:d}"
            # " --workdir={hostname:}:{dir_rho:}execution_output/"  # defines output directions
            " --workdir={dir_rho:}execution_output/"  # defines output directions
            " --job-name=\"RHO{id_data:d}_B{basis_size:d}_S{uncoupled_surfaces:d}_N{uncoupled_modes:d}\""
            " --output=\"{dir_rho:}execution_output/RHO{id_data:d}_B{basis_size:d}_S{uncoupled_surfaces:d}_N{uncoupled_modes:d}.o%A\""
            " {wait_param:s}"  # optional wait parameter
            " --export="
            "\"MODES={uncoupled_modes:d}\""
            ",\"SURFACES={uncoupled_surfaces:d}\""
            ",\"BASIS_SIZE={basis_size:d}\""
            ",\"NUM_OF_TEMPS={n_temps:d}\""
            ",\"ROOT_DIR={data_rho:}\""
            ",\"NUMBER_OF_CORES={n_cpus:d}\""
            ",\"MEMORY_RESERVED={memory:}\""
            ",\"id_data={id_data:}\""
            ",\"id_rho={id_rho:}\""
            ",\"BLAS_MODULE_DIR=/home/ngraymon/dev/privatemodules/openBLAS\""
            ",\"Q_HOSTNAME={hostname:}\""
            " ./server_scripts/rho_job.sh"
            )


def submit_sos_job(FS, param_dict):
    """x"""

    A, N = vIO.extract_dimensions_of_coupled_model(FS=FS)
    param_dict["coupled_modes"] = N
    param_dict["coupled_surfs"] = A

    command = sos_cmd.format(**param_dict)

    # use subprocess to capture the jobid in the return value from SLURM
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, error = p.communicate()

    # this is highly dependent on the output from SLURM and the setup on our local cluster
    # TODO - create a more resilient way of extracting the job id when submitting
    id_sos_job = int(out.decode()[20:])

    # modify the wait arg so that each rho job waits until the sos job has finished
    param_dict["wait_param"] = "--dependency afterok:{:}".format(id_sos_job)
    return


def estimate_memory_usuage(A, N, B):
    """x"""

    # the memory is heavily dependent on the BASIS SIZE
    # 8 bytes for a double, times the number of doubles in the largest matrix
    chosen_memory_size = 8 * pow((A * pow(B, N)), 2)
    # multiply by a factor of 10 for saftey ( we assume at least 4-5 matricies in use at once )
    chosen_memory_size *= 10
    # express the result in a number of GB
    chosen_memory_size /= GB_per_byte
    # make sure we request at least 1 GB of memory
    chosen_memory_size = 1 if (round(chosen_memory_size) is 0) else round(chosen_memory_size)

    # both integers are in units of GB
    assert chosen_memory_size < maximum_memory_per_node, f"Asked for {chosen_memory_size:d}GB"

    return chosen_memory_size


def submit_rho_job(FS=None, path_root=None, id_data=None, id_rho=None, recalculate_sos_file=False, param_dict=None):
    """ recalculate_sos_file is a boolean flag which forces all sos files to be replaced when true, should only use this if the model has changed or debugging"""

    if FS is None:
        assert path_root is not None and type(id_data) is str
        assert id_data is not None and type(id_data) is int
        assert id_rho is not None and type(id_rho) is int  # maybe not necessary?
        FS = file_structure.FileStructure(path_root, id_data, id_rho=id_rho)

    if param_dict is None:
        # or maybe we could look for a parameter dictionary in the current directory?
        param_dict = {"basis_size": 20,
                      "wait_param": "",
                      "n_cpus": 12,
                      "dir_data": FS.path_data,
                      "dir_rho": FS.path_rho,
                      "id_data": FS.id_data,
                      "id_rho": FS.id_rho,
                      "hostname": socket.gethostname(),
                      }

    # read in the model parameters from the JSON files
    A, N = vIO.extract_dimensions_of_sampling_model(FS=FS)
    param_dict["uncoupled_modes"] = N
    param_dict["uncoupled_surfaces"] = A

    # should be able to process more than 2 mode, however this will stay here until a specific multi-mode test case is constructed for saftey
    assert(N == 1 or N == 2)

    # we read in the appropriate parameters from somewhere?
    # source_file = "./model_parameters_source.txt"
    # parameter_dictionary = vIO.parse_model_params(source_file)

    # temperature_list = parameter_dictionary["temperature_list"]
    temperature_list = [300]
    param_dict["n_temps"] = len(temperature_list)
    param_dict["memory"] = estimate_memory_usuage(A, N, param_dict["basis_size"])

    # the reference file
    path = FS.template_sos_rho.format(B=param_dict["basis_size"])

    # if the reference file doesn't exist we need to run SOS to generate it!
    if recalculate_sos_file or not os.path.isfile(path):
        # need to pass param_dict byRef so that wait_arg can be modified if need be
        submit_sos_job(FS, param_dict)

    # otherwise we can just procced to create our rho file
    command = rho_cmd.format(**param_dict)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, error = p.communicate()

    if not error.decode() == "":
        # should this be an exception?
        raise Exception("rho output\n{:s}\n{:s}\n".format(out.decode(), error.decode()))

    return


if (__name__ == "__main__"):
    # set this up to run from the command line in the future?

    # assert(len(sys.argv) == 4)
    # assert(sys.argv[1].isnumeric() and int(sys.argv[1]) >= 1)
    # assert(sys.argv[2].isnumeric() and int(sys.argv[1]) >= 0)
    # assert(sys.argv[3].isnumeric() and int(sys.argv[1]) >= 1)
    pass
