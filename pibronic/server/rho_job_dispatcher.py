# submission script for jobs on nlogn
import numpy as np
import sys, os, socket, subprocess
from hurry.filesize import size, si
import pibronic.data.vibronic_model_io as vIO

# use this flag to force all SOS files to be replaced
# should only be used if the model is changed or there is a heisenbug
FORCE_REPLACE = False

assert(len(sys.argv) == 4)
assert(sys.argv[1].isnumeric() and int(sys.argv[1]) >= 1)
assert(sys.argv[2].isnumeric() and int(sys.argv[1]) >= 0)
assert(sys.argv[3].isnumeric() and int(sys.argv[1]) >= 1)

# workspace_dir = "/home/ngraymon/thesis_code/pimc/workspace/"
workspace_dir = "/work/ngraymon/pimc/"
data_set_id   = int(sys.argv[1])
data_set_dir  = "data_set_{:d}/".format(data_set_id)
rho_set_id    = int(sys.argv[2])
rho_set_dir   = "rho_{:d}/".format(rho_set_id)

# read in the model parameters from the vibronic_model_dictionary.JSON file
number_of_modes, number_of_surfaces = vIO.get_nmode_nsurf_from_sampling_model(data_set_id, rho_set_id)
coupled_modes, coupled_surfs = vIO.get_nmode_nsurf_from_coupled_model(data_set_id)
# currently can only process one or two mode models
assert(number_of_modes == 1 or number_of_modes == 2)

# we read all appropriate parameters from the model_parameters_source.txt
source_file = "./model_parameters_source.txt"
parameter_dictionary = vIO.parse_model_params(source_file)
# temperature_list = parameter_dictionary["temperature_list"]
temperature_list = [300]

# the memory is heavily dependent on the BASIS SIZE
SIZE_OF_GB_IN_BYTES = 1e9
chosen_size_of_basis = int(sys.argv[3])
cores_requested = 12
# 8 bytes for a double, times the number of doubles in the largest matrix
chosen_memory_size = 8 * pow((number_of_surfaces * pow(chosen_size_of_basis, number_of_modes)), 2)
# multiply by a factor of 10 for saftey ( we assume at least 4-5 matricies in use at once)
chosen_memory_size *= 15
# express the result in a number of GB
chosen_memory_size /= SIZE_OF_GB_IN_BYTES
# make sure we request at least 1 GB of memory
chosen_memory_size = 1 if (round(chosen_memory_size) is 0) else round(chosen_memory_size)
# chosen_memory_size = 15

MAXIMUM_NODE_MEMORY = 128
# both integers are in units of GB
assert (chosen_memory_size < MAXIMUM_NODE_MEMORY), "Asked for {:d}GB".format(chosen_memory_size)

# the main command
hostname = socket.gethostname()
sos_qsub = "sbatch" + (  " -m n"  # this stops all mail from being sent
                     + " --priority 0"  # this defines the priority of the job, default is 0
                     + " --mem={memory:}G"
                     # + " --mem-per-cpu={memory:}G"
                     + " --ntasks=1"
                     + " --cpus-per-task={n_cpus:d}"
                     # + " --workdir={:}".format(hostname) + ":{root_dir:}execution_output/"  # defines output directions
                     + " --workdir={root_dir:}execution_output/"  # defines output directions
                     + " --job-name=\"SOS{data_set_id:d}_B{basis_size:d}_S{n_surfaces:d}_N{n_modes:d}\""
                     + " --output=\"{root_dir:}execution_output/SOS{data_set_id:d}_B{basis_size:d}_S{n_surfaces:d}_N{n_modes:d}.o%A\""
                     + " --export="
                     + "\"MODES={n_modes:d}\""
                     + ",\"SURFACES={n_surfaces:d}\""
                     + ",\"BASIS_SIZE={basis_size:d}\""
                     + ",\"NUM_OF_TEMPS={n_temps:d}\""
                     + ",\"ROOT_DIR={root_dir:}\""
                     + ",\"NUMBER_OF_CORES={n_cpus:d}\""
                     + ",\"MEMORY_RESERVED={memory:}\""
                     + ",\"DATA_SET_ID={data_set_id:}\""
                     + ",\"BLAS_MODULE_DIR=/home/ngraymon/dev/privatemodules/openBLAS\""
                     + ",\"Q_HOSTNAME={:}\"".format(hostname)
                     + " ./server_scripts/sos_job.sh"
                    )

# MAX_JOBS = len(bead_list)

# array_qsub  = "sbatch" + (""
#                  # + " --output={:}".format(hostname) + ":{root_dir:}execution_output/" # defines output directions
#                  + " --job-name=\"RHO{data_set_id:d}_B{basis_size:d}_S{n_surfaces:d}_N{n_modes:d}\"" # name the jobs
#                  + " --array=0-{:d}%1".format(MAX_JOBS) # this creates a job array with MAX_JOBS number of jobs
#                  + " --cpu_per_task=16"
#                  + " -N1"
#                  # + " --ntasks=1"
#                  # + " --label"
#                  + " --mem-per-cpu=1000"
#                  # + " -n=5"
#                  + " ./server_scripts/slurm/pimc_job.sh"
#                 )

rho_qsub = "sbatch" + (" -m n"  # this stops all mail from being sent
                     + " --priority 0"  # this defines the priority of the job, default is 0
                     + " --mem={memory:}G"
                     # + " --mem-per-cpu={memory:}G"
                     + " --ntasks=1"
                     + " --cpus-per-task={n_cpus:d}"
                     # + " --workdir={:}".format(hostname) + ":{root_dir:}execution_output/"  # defines output directions
                     + " --workdir={root_dir:}execution_output/"  # defines output directions
                     + " --job-name=\"RHO{data_set_id:d}_B{basis_size:d}_S{n_surfaces:d}_N{n_modes:d}\""
                     + " --output=\"{root_dir:}execution_output/RHO{data_set_id:d}_B{basis_size:d}_S{n_surfaces:d}_N{n_modes:d}.o%A\""
                     + " {wait_param:s}" # optional wait parameter
                     + " --export="
                     + "\"MODES={n_modes:d}\""
                     + ",\"SURFACES={n_surfaces:d}\""
                     + ",\"BASIS_SIZE={basis_size:d}\""
                     + ",\"NUM_OF_TEMPS={n_temps:d}\""
                     + ",\"ROOT_DIR={root_dir:}\""
                     + ",\"NUMBER_OF_CORES={n_cpus:d}\""
                     + ",\"MEMORY_RESERVED={memory:}\""
                     + ",\"DATA_SET_ID={data_set_id:}\""
                     + ",\"RHO_SET_ID={rho_set_id:}\""
                     + ",\"BLAS_MODULE_DIR=/home/ngraymon/dev/privatemodules/openBLAS\""
                     + ",\"Q_HOSTNAME={:}\"".format(hostname)
                     + " ./server_scripts/rho_job.sh"
                    )

# the reference file
root_sos_file = workspace_dir+data_set_dir+"parameters/sos_B{:d}.json".format(chosen_size_of_basis)

wait_arg = "" # empty value
# if the reference file doesn't exist we need to run SOS to generate it!
if FORCE_REPLACE or not os.path.isfile(root_sos_file):
    command = sos_qsub.format(
                n_modes=coupled_modes,
                n_surfaces=coupled_surfs,
                basis_size=chosen_size_of_basis,
                n_temps=len(temperature_list),
                n_cpus=cores_requested,
                memory=chosen_memory_size,
                root_dir=workspace_dir+data_set_dir,
                data_set_id=data_set_id
                )
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, error = p.communicate()
    # print("SOS FIRST", out.decode(), "\n", error.decode())
    SOS_JOB_ID = int(out.decode()[20:])
    wait_arg = "--dependency afterok:{:}".format(SOS_JOB_ID)

# otherwise we can just procced to create our rho file
command = rho_qsub.format(
            wait_param=wait_arg,
            n_modes=number_of_modes,
            n_surfaces=number_of_surfaces,
            basis_size=chosen_size_of_basis,
            n_temps=len(temperature_list),
            n_cpus=cores_requested,
            memory=chosen_memory_size,
            root_dir=workspace_dir+data_set_dir+rho_set_dir,
            data_set_id=data_set_id,
            rho_set_id=rho_set_id
            )
p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
out, error = p.communicate()
if not error.decode() == "":
    print("RHO AFTER", out.decode(), "\n", error.decode())
