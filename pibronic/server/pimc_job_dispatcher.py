# submission script for jobs on nlogn
import numpy as np
import sys, os, socket, subprocess
import pibronic.data.vibronic_model_io as vIO

assert(len(sys.argv) == 3)
assert(sys.argv[1].isnumeric() and int(sys.argv[1]) >= 1)
assert(sys.argv[2].isnumeric() and int(sys.argv[1]) >= 0)

# workspace_dir = "/home/ngraymon/thesis_code/pimc/workspace/"
workspace_dir = "/work/ngraymon/pimc/"
data_set_id   = int(sys.argv[1])
data_set_dir  = "data_set_{:d}/".format(data_set_id)
rho_set_id    = int(sys.argv[2])
rho_set_dir   = "rho_{:d}/".format(rho_set_id)

directory_path = workspace_dir + data_set_dir
sampling_path = directory_path + rho_set_dir
assert(os.path.exists(sampling_path))

# unsafe
# os.system("rm -r {:}*".format(workspace_dir+data_set_dir + "execution_output/"))
# os.system("rm -r {:}*".format(workspace_dir+data_set_dir + "results/"))
# os.system("rm -r {:}*".format(workspace_dir+data_set_dir + "output/"))
# os.system("rm -r {:}*".format(workspace_dir+data_set_dir + "plots/"))



# read in the model parameters from the vibronic_model_dictionary.JSON file
coupled_modes, coupled_surfaces = vIO.get_nmode_nsurf_from_coupled_model(data_set_id)
rho_modes, rho_surfaces = vIO.get_nmode_nsurf_from_sampling_model(data_set_id, rho_set_id)

# we read all appropriate parameters from the model_parameters_source.txt
source_file = "./model_parameters_source.txt"
parameter_dictionary = vIO.parse_model_params(source_file)
temperature_list     = parameter_dictionary["temperature_list"]
# sample_list          = parameter_dictionary["sample_list"] # we dont need this anymore?
# bead_list            = parameter_dictionary["bead_list"]
# bead_list            = np.array([170])
# bead_list            = np.array([1,4,8,16,32,64,128])
# bead_list            = np.array([3])
# bead_list            = np.sort(np.array([1,4,8,16,32,64,128] + [2, 6, 10, 12, 14, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]))
# bead_list            = np.sort(np.array([1,4,8,16,32,64,128] + [2, 6, 10, 12, 14, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]))
# bead_list            = np.array([7])
# bead_list = np.sort(np.array([16,32,64,128] + [12, 14, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]))
# memory_list          = np.ones_like(bead_list, dtype=int) + (bead_list // 20)

# bead_list = np.sort(np.array([16,32,64,128] + [12, 14, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]))
# bead_list = np.array([1, 4, 8])
# memory_list = np.where(bead_list < 50, [10], [20])
# memory_list = np.where(bead_list < 50, [10], [20])
# print(bead_list)
# print(memory_list)
# sys.exit(0)


bead_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 75, 100])
memory_list = np.array([25]*len(bead_list))

# choose these runtime parameters
number_of_samples = int(1e6)
block_size = int(1e2)
number_of_cpus = 1

# temporary testing
temperature_list = [300]


hostname = socket.gethostname()
sbatch = "sbatch" + (
                       " -m n"  # this stops all mail from being sent
                     + " --priority 0"  # this defines the priority of the job, default is 0
                     + " --mem={memory:}G"
                     + " --partition=highmem"
                     + " --ntasks=1"
                     + " --cpus-per-task={n_cpus:d}"
                     + " --workdir={:}".format(hostname) + ":{rho_dir:}execution_output/"  # defines output directions
                     + " --job-name=\"F{data_set_id:}_A{cA:d}_N{cN:d}_X{n_samples:d}_P{n_beads:d}_T{temp:d}\""
                     + " --output=\"{rho_dir:}execution_output/\"F{data_set_id:}_A{cA:d}_N{cN:d}_X{n_samples:d}_P{n_beads:d}_T{temp:d}\".o%A\""
                     + " {wait_param:s}" # optional wait parameter
                     + " --export="
                     + "\"MODES={cN:d}\""
                     + ",\"SURFACES={cA:d}\""
                     + ",\"RHO_MODES={rN:d}\""
                     + ",\"RHO_SURFACES={rA:d}\""
                     + ",\"SAMPLES={n_samples:d}\""
                     + ",\"BLOCKS={block_size:d}\""
                     + ",\"BEADS={n_beads:d}\""
                     + ",\"TEMP={temp:d}\""
                     + ",\"DELTA_BETA={delta_beta:f}\""
                     + ",\"ROOT_DIR={root_dir:}\""
                     + ",\"RHO_DIR={rho_dir:}\""
                     + ",\"NUMBER_OF_CORES={n_cpus:d}\""
                     + ",\"MEMORY_RESERVED={memory:}\""
                     + ",\"DATA_SET_ID={data_set_id:}\""
                     + ",\"RHO_SET_ID={rho_set_id:}\""
                     + ",\"PYTHON3_PATH=/home/ngraymon/dev/ubuntu/16.04/bin/python3\""
                     + ",\"Q_HOSTNAME={:}\"".format(hostname)
                     + " ./server_scripts/pimc_job.sh"
                    )


if(False):
    os.system(sbatch.format(
        cN=coupled_modes,
        cA=coupled_surfaces,
        rN=rho_modes,
        rA=rho_surfaces,
        n_samples=int(1e6),
        block_size=int(1e2),
        n_beads=7,
        n_cpus=1,
        delta_beta=2.0e-4,
        temp=300,
        memory=3, #GB
        root_dir=directory_path,
        rho_dir=sampling_path,
        data_set_id=data_set_id,
        rho_set_id=rho_set_id,
        wait_param="", # blank no waiting
        )
    )
    sys.exit(0)

# store one job ID for each value of P
JOB_ARRAY = [0]*len(bead_list)


if(hostname == "nlogn"):
    for bead_index, bead_val in enumerate(bead_list):
        command = sbatch.format(
                cN=coupled_modes,
                cA=coupled_surfaces,
                rN=rho_modes,
                rA=rho_surfaces,
                n_samples=number_of_samples,
                block_size=block_size,
                n_beads=bead_val,
                n_cpus=number_of_cpus,
                delta_beta=2.0e-4,
                temp=temperature_list[-1],
                memory=memory_list[bead_index],
                root_dir=directory_path,
                rho_dir=sampling_path,
                data_set_id=data_set_id,
                rho_set_id=rho_set_id,
                wait_param="", # blank - no waiting
                )
        print(command)
        sys.exit(0)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, error = p.communicate()
        JOB_ARRAY[bead_index] = int(out.decode()[20:])

        for temp_val in temperature_list[-2::-1]:  # go backwards because higher temps are easier
            command = sbatch.format(
                    cN=coupled_modes,
                    cA=coupled_surfaces,
                    rN=rho_modes,
                    rA=rho_surfaces,
                    n_samples=number_of_samples,
                    block_size=block_size,
                    n_beads=bead_val,
                    n_cpus=number_of_cpus,
                    delta_beta=2.0e-4,
                    temp=temp_val,
                    memory=memory_list[bead_index],
                    root_dir=directory_path,
                    rho_dir=sampling_path,
                    data_set_id=data_set_id,
                    rho_set_id=rho_set_id,
                    wait_param="--dependency afterok:{:}".format(JOB_ARRAY[bead_index]),
                    )
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, error = p.communicate()
            JOB_ARRAY[bead_index] = int(out.decode()[20:])

elif(hostname=="feynman"):
    for bead_index, bead_val in enumerate(bead_list):
    # for bead_index, bead_val in enumerate([400]):
        # for temp_val in [300]:
        for temp_val in temperature_list:
            command = sbatch.format(
                    cN=coupled_modes,
                    cA=coupled_surfaces,
                    rN=rho_modes,
                    rA=rho_surfaces,
                    n_samples=number_of_samples,
                    block_size=block_size,
                    n_beads=bead_val,
                    n_cpus=number_of_cpus,
                    delta_beta=2.0e-4,
                    temp=temp_val,
                    memory=memory_list[bead_index],
                    root_dir=directory_path,
                    rho_dir=sampling_path,
                    data_set_id=data_set_id,
                    rho_set_id=rho_set_id,
                    wait_param="", # blank no waiting
                    )
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, error = p.communicate()





