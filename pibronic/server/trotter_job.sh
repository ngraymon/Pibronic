#!/bin/bash
# submit script for running a single job on nlogn
#
#
################################ SETUP AND USER INFO ################################
#

# define file paths and other constants
SCRATCH_DIR="/scratch/$USER/pimc/data_set_${DATA_SET_ID}/parameters"
INPUT_DIR="${ROOT_DIR}parameters/"
JOB_ID=${SLURM_JOBID%".nlogn.nlogn"} # remove the nlogn part (if present)
JOB_ID=${SLURM_JOBID%".feynman"}		   # remove the feynman part (if present)
TROTTER_SCRIPT="$SLURM_SUBMIT_DIR/workspace/fast_hacked.exe";

# tell user where scratch directory is
echo ${SCRATCH_DIR}
# make sure scrath directory exists
mkdir -p ${SCRATCH_DIR}

# some user feedback
echo "Running job ${JOB_ID} on ${HOSTNAME} with ${SLURM_CPUS_PER_TASK} cores and ${MEMORY_RESERVED}GB of memory"
echo "Data set ${DATA_SET_ID} has ${SURFACES} electronic surfaces and ${MODES} normal modes"
echo "This execution run is diagonalizing the Hamiltonian with a basis set of size ${BASIS_SIZE} and ${BEADS} beads"
echo "input directory: (${INPUT_DIR})"
echo "root directory: (${ROOT_DIR})"
echo "jobname: $SLURM_JOB_NAME"

#
################################ Meat & Potatoes ################################
#

# load libraries for openblas
module load ${BLAS_MODULE_DIR}
# set the number of cores to be used
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
# print the exact code we are about to run
echo ${TROTTER_SCRIPT} "-T" ${BASIS_SIZE} ${SURFACES} ${MODES} ${BEADS} ${NUM_OF_TEMPS} ${DATA_SET_ID}
# execute the job
${TROTTER_SCRIPT} "-T" ${BASIS_SIZE} ${SURFACES} ${MODES} ${BEADS} ${NUM_OF_TEMPS} ${DATA_SET_ID}

# retrive useful information about the job's execution and then print that info
EXECSTATS=$(ssh ${Q_HOSTNAME} sacct -j $SLURM_JOBID --format=CPUTime,CPUTimeRAW,AveVMSize)

echo "Execution stats: "
echo ${EXECSTATS}
echo "if there are no values it means the job didn't run long enough to record values"

# copy the files from scratch to the ROOT_DIR (most likely /work/)
mv --force ${SCRATCH_DIR}/trotter_P${BEADS}_B${BASIS_SIZE}.json ${ROOT_DIR}parameters/

# some user feedback
echo "Output file was here: $HOSTNAME:${SCRATCH_DIR}/"
echo "Output files should be here now: ${ROOT_DIR}parameters/"

# finish up
cd "$SLURM_SUBMIT_DIR"
echo Job Script Finished Successfully
#
#
#
#
####