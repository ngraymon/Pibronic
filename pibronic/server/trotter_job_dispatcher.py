""" submission script for trotter jobs on a server"""

# system imports
import subprocess
import socket
import sys

# third party imports

# local imports
from ..vibronic import vIO

hostname = socket.gethostname()
sos_qsub = "sbatch" + (
                     " -m n"  # this stops all mail from being sent
                     + " --priority 0"  # this defines the priority of the job, default is 0
                     + " --mem={memory:}G"
                     + " --ntasks=1"
                     + " --cpus-per-task={n_cpus:d}"
                     # + " -D {:}".format(hostname) + ":{root_dir:}execution_output/"  # defines output directions
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

trotter_qsub = "sbatch" + (
                     " -m n"  # this stops all mail from being sent
                     + " --priority 0"  # this defines the priority of the job, default is 0
                     + " --mem={memory:}G"
                     + " --ntasks=1"
                     + " --cpus-per-task={n_cpus:d}"
                     # + " -D {:}".format(hostname) + ":{root_dir:}execution_output/"  # defines output directions
                     + " --workdir={root_dir:}execution_output/"  # defines output directions
                     + " --job-name=\"T{data_set_id:d}_B{basis_size:d}_S{n_surfaces:d}_N{n_modes:d}_P{n_beads:d}\""
                     + " --output=\"{root_dir:}execution_output/T{data_set_id:d}_B{basis_size:d}_S{n_surfaces:d}_N{n_modes:d}_P{n_beads:d}.o%A\""
                     + " --dependency afterok:{wait_id:}" # this prevents the job from starting until another job has finished
                     + " --export="
                     + "\"BEADS={n_beads:d}\""
                     + ",\"MODES={n_modes:d}\""
                     + ",\"SURFACES={n_surfaces:d}\""
                     + ",\"BASIS_SIZE={basis_size:d}\""
                     + ",\"NUM_OF_TEMPS={n_temps:d}\""
                     + ",\"ROOT_DIR={root_dir:}\""
                     + ",\"NUMBER_OF_CORES={n_cpus:d}\""
                     + ",\"MEMORY_RESERVED={memory:}\""
                     + ",\"DATA_SET_ID={data_set_id:}\""
                     + ",\"BLAS_MODULE_DIR=/home/ngraymon/dev/privatemodules/openBLAS\""
                     + ",\"Q_HOSTNAME={:}\"".format(hostname)
                     + " ./server_scripts/trotter_job.sh"
                    )

# MAX_JOBS = len(bead_list)

# array_qsub  = "sbatch" + (""
#                  # + " --output={:}".format(hostname) + ":{root_dir:}execution_output/" # defines output directions
#                  + " --job-name=\"T{data_set_id:d}_B{basis_size:d}_S{n_surfaces:d}_N{n_modes:d}\"" # name the job array
#                  + " --array=0-{:d}%1".format(MAX_JOBS) # this creates a job array with MAX_JOBS number of jobs
#                  + " --cpus-per-task={:d}".format(cores_requested)
#                  + " -N1"
#                  # + " --ntasks=1"
#                  # + " --label"
#                  # + " --mem-per-cpu=1000"
#                  # + " -n=5"
#                  + " ./server_scripts/trotter_job.sh"
#                 )

# alter_qsub = "scontrol" + (" update"
#                  + " ArrayJobId={job_id:}"
#                  + " UserId=ngraymon"
#                  # + " --mem={memory:}GB -n {n_cpus:d}"
#                  + " --export="
#                  + ",\"BEADS={n_beads:d}\""
#                  + " \"MODES={n_modes:d}\""
#                  + ",\"SURFACES={n_surfaces:d}\""
#                  + ",\"BASIS_SIZE={basis_size:d}\""
#                  + ",\"NUM_OF_TEMPS={n_temps:d}\""
#                  + ",\"ROOT_DIR={root_dir:}\""
#                  + ",\"NUMBER_OF_CORES={n_cpus:d}\""
#                  + ",\"MEMORY_RESERVED={memory:}\""
#                  + ",\"DATA_SET_ID={data_set_id:}\""
#                  + ",\"BLAS_MODULE_DIR=/home/ngraymon/dev/privatemodules/openBLAS\""
#                  + ",\"Q_HOSTNAME={:}\"".format(hostname)
#                 )


if (__name__ == "__main__"):

    assert(len(sys.argv) == 3)
    assert(sys.argv[1].isnumeric() and int(sys.argv[1]) >= 1)
    assert(sys.argv[2].isnumeric() and int(sys.argv[1]) >= 1)

    # workspace_dir = "/home/ngraymon/thesis_code/pimc/workspace/"
    workspace_dir = "/work/ngraymon/pimc/"
    data_set_id = int(sys.argv[1])
    data_set_dir = "data_set_{:d}/".format(data_set_id)

    # read in the model parameters from the vibronic_model_dictionary.JSON file
    number_of_surfaces, number_of_modes = vIO.extract_dimensions_of_coupled_model(data_set_id)
    # currently can only process one or two mode models
    assert(number_of_modes == 1 or number_of_modes == 2)

    # we read all appropriate parameters from the model_parameters_source.txt
    source_file = "./model_parameters_source.txt"
    parameter_dictionary = vIO.parse_model_params(source_file)
    # temperature_list = parameter_dictionary["temperature_list"]
    temperature_list = [300]
    # bead_list = parameter_dictionary["bead_list"]
    # already_list = np.array([100, 10, 120, 12, 140, 14, 160, 16, 180, 18, 200, 20, 22, 24, 250, 26, 28, 2, 300, 30, 32, 34, 350, 36, 38, 400, 40, 44, 48, 4, 52, 56, 60, 64, 68, 6, 72, 76, 80, 8, 90])
    # bead_list = np.array([p for p in bead_list if p not in already_list])
    #
    # already_list = [1,4,8,12,16,24,32,40,50,64,80,100,128]
    # already_list = np.sort([30,16,20,12,14,25,18,35,32,40,45,50,55,65,60,70,64,80,75,85,95,90,100])
    # bead_list = np.array([i for i in range(1,101,1) if i not in already_list])

    # bead_list = [2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, ]

    # bead_list = [250, 300]
    bead_list = [120, 140, 160, 180, 200, 250, 300]

    # the memory is heavily dependent on the BASIS SIZE
    SIZE_OF_GB_IN_BYTES = 1e9
    chosen_size_of_basis = int(sys.argv[2])
    cores_requested = 16
    # 8 bytes for a double, times the number of doubles in the largest matrix
    chosen_memory_size = 8 * pow((number_of_surfaces * pow(chosen_size_of_basis, number_of_modes)), 2)
    # multiply by a factor of 10 for saftey ( we assume at least 4-5 matricies in use at once)
    chosen_memory_size *= 15
    # express the result in a number of GB
    chosen_memory_size /= SIZE_OF_GB_IN_BYTES
    # make sure we request at least 1 GB of memory
    chosen_memory_size = 1 if (round(chosen_memory_size) is 0) else round(chosen_memory_size)

    MAXIMUM_NODE_MEMORY = 128
    # both integers are in units of GB
    assert (chosen_memory_size < MAXIMUM_NODE_MEMORY), "Asked for {:d}GB".format(chosen_memory_size)

    # flag for debugging
    JUST_SOS = False

    SOS_JOB_ID = 0
    command = sos_qsub.format(
                    n_modes=number_of_modes,
                    n_surfaces=number_of_surfaces,
                    basis_size=chosen_size_of_basis,
                    n_temps=len(temperature_list),
                    n_cpus=cores_requested,
                    memory=chosen_memory_size,
                    root_dir=workspace_dir+data_set_dir,
                    data_set_id=data_set_id
                    )

    if(JUST_SOS):
        print(command)

    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, error = p.communicate()

    if(JUST_SOS):
        sys.exit(0)

    # we only want the number
    SOS_JOB_ID = int(out.decode()[20:])
    for bead_index, bead_val in enumerate(bead_list):
        command = trotter_qsub.format(
                wait_id=SOS_JOB_ID,
                n_modes=number_of_modes,
                n_surfaces=number_of_surfaces,
                basis_size=chosen_size_of_basis,
                n_beads=bead_val,
                n_temps=len(temperature_list),
                n_cpus=cores_requested,
                memory=chosen_memory_size,
                root_dir=workspace_dir+data_set_dir,
                data_set_id=data_set_id
                )
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, error = p.communicate()

    # # we only want the number
    # SOS_JOB_ID = out.split(sep=b'.', maxsplit=1)[0].decode()

    # command = array_qsub.format(
    #         data_set_id=data_set_id,
    #         n_modes=number_of_modes,
    #         n_surfaces=number_of_surfaces,
    #         basis_size=chosen_size_of_basis,
    #         )
    # p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # out, error = p.communicate()
    # JOB_ARRAY_ROOT = int(out.decode()[20:])

    # for bead_index, bead_val in enumerate(bead_list):
    #     command = alter_qsub.format(
    #             wait_id=SOS_JOB_ID,
    #             n_modes=number_of_modes,
    #             n_surfaces=number_of_surfaces,
    #             basis_size=chosen_size_of_basis,
    #             n_beads=bead_val,
    #             n_temps=len(temperature_list),
    #             n_cpus=cores_requested,
    #             memory=chosen_memory_size,
    #             root_dir=workspace_dir+data_set_dir,
    #             data_set_id=data_set_id,
    #             job_id=JOB_ARRAY_ROOT+bead_index,
    #             )
    #     p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    #     out, error = p.communicate()
    #     print(out.decode(),error.decode())
