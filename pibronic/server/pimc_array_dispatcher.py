"""x"""

# submission script for jobs on nlogn
import numpy as np
import sys, os, socket, subprocess
import pibronic.data.vibronic_model_io as vIO


if (__name__ == "__main__"):
    pass

    assert(len(sys.argv) == 2)
    assert(sys.argv[1].isnumeric() and int(sys.argv[1]) >= 1)

    #workspace_dir = "/home/ngraymon/thesis_code/pimc/workspace/"
    workspace_dir = "/work/ngraymon/pimc/"
    data_set_id = int(sys.argv[1])
    data_set_dir  = "data_set_{:d}/".format(data_set_id)

    # unsafe
    # os.system("rm -r {:}*".format(workspace_dir+data_set_dir + "execution_output/"))
    # os.system("rm -r {:}*".format(workspace_dir+data_set_dir + "results/"))
    # os.system("rm -r {:}*".format(workspace_dir+data_set_dir + "output/"))
    # os.system("rm -r {:}*".format(workspace_dir+data_set_dir + "plots/"))

    # choose these runtime parameters
    X_samples = int(1e6)
    block_size = int(1e2)
    number_of_cpus = 1
    delta_beta = 2.0E-4

    # read in the model parameters from the vibronic_model_dictionary.JSON file
    number_of_modes, number_of_surfaces = vIO.get_nmode_nsurf_from_coupled_model(data_set_id)

    # we read all appropriate parameters from the model_parameters_source.txt
    source_file = "./model_parameters_source.txt"
    parameter_dictionary = vIO.parse_model_params(source_file)
    temperature_list     = parameter_dictionary["temperature_list"]
    # sample_list          = parameter_dictionary["sample_list"] # we dont need this anymore?
    # bead_list = np.sort(np.array([1,4,8,16,32,64,128] + [2, 6, 10, 12, 14, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]))
    bead_list = np.array([4, 8, 16, 32, 64])
    memory_list          = np.ones_like(bead_list, dtype=int) + (bead_list // 20)


    hostname = socket.gethostname()

    MAX_JOBS = len(temperature_list)*len(bead_list)

    array_qsub  = "sbatch" + (""
                     # + " --output={:}".format(hostname) + ":{root_dir:}execution_output/" # defines output directions
                     + " --job-name=\"F{:}".format(data_set_id) + "_S{n_surfaces:d}_N{n_modes:d}\"" # name the jobs
                     + " --array=0-{:d}%1".format(MAX_JOBS) # this creates a job array with MAX_JOBS number of jobs
                     # + " --cpu-per-task=1"
                     + " -N1"
                     # + " --ntasks=1"
                     # + " --label"
                     + " --mem-per-cpu=1000"
                     # + " -n=5"
                     + " ./server_scripts/slurm/pimc_job.sh"
                    )


    # + " -o {:}".format(hostname) + ":{root_dir:}execution_output/" # defines output directions
    # + " -N \"F{:}".format(data_set_id) + "_A{n_surfaces:d}_N{n_modes:d}_X{n_samples:d}_P{n_beads:d}_T{temp:d}\""

    alter_qsub = "scontrol" + (" update"
                     + " ArrayJobId={job_id:}"
                     + " UserId=ngraymon"
                     # + " --mem={memory:}GB -n {n_cpus:d}"
                     + " --export="
                     + " \"MODES={n_modes:d}\""
                     + ",\"SURFACES={n_surfaces:d}\""
                     + ",\"SAMPLES={n_samples:d}\""
                     + ",\"BLOCKS={block_size:d}\""
                     + ",\"BEADS={n_beads:d}\""
                     + ",\"TEMP={temp:d}\""
                     + ",\"DELTA_BETA={delta_beta:f}\""
                     + ",\"ROOT_DIR={root_dir:}\""
                     + ",\"DATA_SET_ID={data_set_id:}\""
                     + ",\"PYTHON3_PATH=/home/ngraymon/dev/ubuntu/16.04/bin/python3\""
                     + ",\"Q_HOSTNAME={:}\"".format(hostname)
                     + ",\"NUMBER_OF_CORES={n_cpus:d}\""
                     + ",\"MEMORY_RESERVED={memory:}\""
                    )
    #ArrayTaskThrottle=5

    # submit the job array
    command = array_qsub.format(
            n_modes=number_of_modes,
            n_surfaces=number_of_surfaces,
            root_dir=workspace_dir+data_set_dir,
            )
    # obtain the job id root
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, error = p.communicate()
    JOB_ARRAY_ROOT = int(out.decode()[20:])

    for bead_index, bead_val in enumerate(bead_list):
        for temp_index, temp_val in enumerate(temperature_list):
            command = alter_qsub.format(
                    n_modes=number_of_modes,
                    n_surfaces=number_of_surfaces,
                    n_samples=X_samples,
                    block_size=block_size,
                    n_cpus=number_of_cpus,
                    delta_beta=delta_beta,
                    n_beads=bead_val,
                    temp=temp_val,
                    memory=memory_list[bead_index],
                    root_dir=workspace_dir+data_set_dir,
                    data_set_id=data_set_id,
                    job_id=JOB_ARRAY_ROOT+temp_index+bead_index*len(temperature_list),
                    )
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, error = p.communicate()
            print(out.decode(),error.decode())
            #JOB_ARRAY_ROOT = out.split(sep=b'.', maxsplit=1)[0].decode()
