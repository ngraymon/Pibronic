""" submission script to automate submitting pimc jobs on slurm server """

# system imports
import subprocess
import socket

# third party imports
import numpy as np

# local imports
from ..vibronic import vIO
from ..data import file_structure
from ..constants import delta_beta


pimc_cmd = "sbatch"
pimc_cmd += (
             " -m n"  # this stops all mail from being sent
             " --priority 0"  # this defines the priority of the job, default is 0
             " --mem={memory:}G"
             " --partition={partition:}"
             " --ntasks=1"
             " --cpus-per-task={n_cpus:d}"
             # this should be factored out into file_structure
             " --workdir={hostname:}:{dir_rho:}execution_output/"  # defines output directions
             # the job name here should be factored out into file_name in data
             " --job-name=\"F{id_data:}_A{coupled_surfaces:d}_N{coupled_modes:d}_X{n_samples:d}_P{n_beads:d}_T{temp:d}\""
             # this should be factored out into file_name and file_structure
             " --output=\"{dir_rho:}execution_output/\"F{id_data:}_A{cA:d}_N{cN:d}_X{n_samples:d}_P{n_beads:d}_T{temp:d}\".o%A\""
             " {wait_param:s}"  # optional wait parameter
             " --export="
             "\"MODES={coupled_modes:d}\""
             ",\"SURFACES={coupled_surfaces:d}\""
             ",\"RHO_MODES={uncoupled_modes:d}\""
             ",\"RHO_SURFACES={uncoupled_surfaces:d}\""
             ",\"SAMPLES={n_samples:d}\""
             ",\"BLOCKS={block_size:d}\""
             ",\"BEADS={n_beads:d}\""
             ",\"TEMP={temp:d}\""
             ",\"DELTA_BETA={delta_beta:f}\""
             ",\"ROOT_DIR={dir_root:}\""
             ",\"RHO_DIR={dir_rho:}\""
             ",\"NUMBER_OF_CORES={n_cpus:d}\""
             ",\"MEMORY_RESERVED={memory:}\""
             ",\"id_data={id_data:}\""
             ",\"id_rho={id_rho:}\""
             # this might be worth eventually removing?
             ",\"PYTHON3_PATH=/home/ngraymon/dev/privatemodules/openBLAS\""
             ",\"Q_HOSTNAME={hostname:}\""
             # this should be made more robust?
             # use an absolute path?
             " ./server_scripts/pimc_job.sh"
             )


def extract_job_id(out, error):
    """this might need to be more dynamic and change with type of server"""
    return int(out.decode()[20:])


def submit_pimc_job_to_nlogn(lst_T, lst_P, lst_mem, param_dict):
    """x"""
    for bead_index, bead_val in enumerate(lst_P):
        param_dict["wait_param"] = ""
        command = pimc_cmd.format(memory=lst_mem[bead_index],
                                  n_beads=bead_val,
                                  temp=lst_T[-1],
                                  **param_dict
                                  )
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, error = p.communicate()
        previous_job_id = extract_job_id(out, error)

        # go backwards because higher temps are easier
        for temp_val in lst_T[-2::-1]:
            param_dict["wait_param"] = f"--dependency afterok:{previous_job_id:d}"
            command = pimc_cmd.format(memory=lst_mem[bead_index],
                                      n_beads=bead_val,
                                      temp=temp_val,
                                      **param_dict
                                      )
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, error = p.communicate()
            previous_job_id = extract_job_id(out, error)
    return


def submit_pimc_job_to_feynman(lst_T, lst_P, lst_mem, param_dict):
    """x"""
    for bead_index, bead_val in enumerate(lst_P):
        for temp_val in lst_T:
            command = pimc_cmd.format(memory=lst_mem[bead_index],
                                      n_beads=bead_val,
                                      temp=temp_val,
                                      **param_dict
                                      )
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, error = p.communicate()
    return


def submit_pimc_job(FS=None, path_root=None, id_data=None, id_rho=None, param_dict=None):
    """x"""

    if FS is None:
        assert path_root is not None and type(id_data) is str
        assert id_data is not None and type(id_data) is int
        assert id_rho is not None and type(id_rho) is int  # maybe not necessary?
        FS = file_structure.FileStructure(path_root, id_data, id_rho=id_rho)

    if param_dict is None:
        # or maybe we could look for a parameter dictionary in the current directory?
        param_dict = {"basis_size": 20,
                      "wait_param": "",
                      "n_cpus": 1,
                      "temperature_list": [300, ],
                      "delta_beta": delta_beta,
                      "n_samples": int(1e6),
                      "block_size": int(1e2),
                      "dir_root": FS.path_root,
                      "dir_data": FS.path_data,
                      "dir_rho": FS.path_rho,
                      "id_data": FS.id_data,
                      "id_rho": FS.id_rho,
                      "hostname": socket.gethostname(),
                      "partition": "serial",  # use "highmem" for large memory jobs
                      }

    # read in the model parameters from the JSON files
    A, N = vIO.extract_dimensions_of_coupled_model(FS=FS)
    param_dict["coupled_modes"] = N
    param_dict["coupled_surfaces"] = A
    # read in the model parameters from the JSON files
    A, N = vIO.extract_dimensions_of_sampling_model(FS=FS)
    param_dict["uncoupled_modes"] = N
    param_dict["uncoupled_surfaces"] = A

    # coupled_modes, coupled_surfaces = vIO.extract_dimensions_of_coupled_model(id_data)
    # rho_surfaces, rho_modes  = vIO.extract_dimensions_of_sampling_model(id_data, id_rho)

    temperature_list = param_dict["temperature_list"]
    # sample_list          = param_dict["sample_list"] # we dont need this anymore?
    # bead_list            = param_dict["bead_list"]
    # memory_list          = np.ones_like(bead_list, dtype=int) + (bead_list // 20)
    # bead_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 75, 100])
    bead_list = np.array([12, ])
    memory_list = np.array([25]*len(bead_list))

    if(param_dict["hostname"] == "nlogn"):
        submit_pimc_job_to_nlogn(temperature_list, bead_list, memory_list, param_dict)
    if(param_dict["hostname"] == "feynman"):
        submit_pimc_job_to_feynman(temperature_list, bead_list, memory_list, param_dict)
    else:
        raise Exception("This server is currently not supported")
    return


if (__name__ == "__main__"):
    # set this up to run from the command line in the future?

    # assert(len(sys.argv) == 3)
    # assert(sys.argv[1].isnumeric() and int(sys.argv[1]) >= 1)
    # assert(sys.argv[2].isnumeric() and int(sys.argv[1]) >= 0)
    pass
