from .context import pibronic
import numpy as np
import sys
import os

assert(len(sys.argv) == 2)
assert(sys.argv[1].isnumeric() and int(sys.argv[1]) >= 1)

workspace_dir = "/home/ngraymon/scp_2016/workspace/"
workspace_dir = "/work/ngraymon/pimc/"
data_set_dir  = "data_set_{:d}/".format(int(sys.argv[1]))

# unsafe
#os.system("rm -r {:}*".format(workspace_dir+data_set_dir + "execution_output/"))
#os.system("rm -r {:}*".format(workspace_dir+data_set_dir + "results/"))
#os.system("rm -r {:}*".format(workspace_dir+data_set_dir + "output/"))
#os.system("rm -r {:}*".format(workspace_dir+data_set_dir + "plots/"))

number_of_surfaces      = None
number_of_modes         = None
temperature_list        = None
sample_list             = None
memory_list             = None
bead_list               = None

# we read all appropriate parameters from the model_parameters_source.txt
filename_source = "./model_parameters_source.txt"

if os.path.exists(filename_source) and os.path.isfile(filename_source):
    with open(filename_source, 'r') as source_file:
        while not source_file.readline() == "":
            header = source_file.readline().strip()
            data = source_file.readline().strip()
            if(header == "quadratic_division_term"):
                continue
            elif(header == "linear_division_term"):
                continue
            elif(header == "frequency_range"):
                continue            
            elif(header == "number_of_surfaces"):
                number_of_surfaces = int(data)
            elif(header == "number_of_modes"):
                number_of_modes = int(data)
            elif(header == "temperature_list"):
                temperature_list = np.fromstring(data, dtype=np.int, sep=',')
            elif(header == "sample_list"):
                sample_list = np.fromstring(data, sep=',').astype(dtype=np.int32)
            elif(header == "memory_list"):
                memory_list = np.fromstring(data, dtype=np.int, sep=',')
            elif(header == "bead_list"):
                bead_list = np.fromstring(data, dtype=np.int, sep=',')
            else:
                raise ValueError("header {:} is not valid\nCheck that your model_parameters_source.txt has the correct formatting".format(header))
else:
    raise FileNotFoundError("Cannot find {}".format(filename_source))

#sample_list = [1e3, 1e4]
#memory_list = [2, 5, 15]
#bead_list           = [5, 10, 15, 20, 25, 35, 45, 60, 80, 100, 120, 140, 160, 180, 200]
#temperature_list    = [40, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 1000, 1500, 2000, 3000]


qsub_string = "qsub" + (
                     " -j oe" # this joins the stderr and stdout
                     + " -m n"  # this stops all mail from being sent
                     + " -p 0"  # this defines the priority of the job, default is 0
                     + " -l mem={memory:}GB:nodes=1:ppn={n_cpus:d}" # requests amount of memory, compute nodes, and processors per node
                     + " -o nlogn:{root_dir:}execution_output/" # defines output directions
                     #+ " -N \"F{:}".format(int(sys.argv[1])) + "_X{n_samples:d}_memory_profiling\""
                     + " -N \"MEM_F{:}".format(int(sys.argv[1])) + "_X{n_samples:d}_S{n_surfaces:d}_N{n_modes:d}_B{n_beads:d}_T{temp:d}\""
                     + " -v"
                     + "\"MODES={n_modes:d}\""
                     + ",\"SURFACES={n_surfaces:d}\""
                     + ",\"SAMPLES={n_samples:d}\""
                     + ",\"BLOCKS={block_size:d}\""
                     + ",\"BEADS={n_beads:d}\""
                     + ",\"TEMP={temp:d}\""
                     + ",\"ROOT_DIR={root_dir:}\""
                     + ",\"NUMBER_OF_CORES={n_cpus:d}\""
                     + ",\"MEMORY_RESERVED={memory:}\""
                     + " profile_script.sh"
                    )



os.system(qsub_string.format(   n_modes=number_of_modes,
	                            n_surfaces=number_of_surfaces,
	                            n_samples=int(1e2),
	                            block_size=int(1e2),
	                            n_beads=45, 
	                            n_cpus=1,
	                            temp=100,
	                            memory=1,
	                            root_dir=workspace_dir+data_set_dir,
	                            ))

