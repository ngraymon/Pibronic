#!/bin/bash
# submit script for running a single job on nlogn
#
################################ SETUP AND USER INFO ################################
#
# SLURM_ARRAY_JOB_ID
# SLURM_ARRAY_TASK_ID
# SLURM_ARRAY_TASK_COUNT
# SLURM_ARRAY_TASK_MAX
# SLURM_ARRAY_TASK_MIN

echo $SLURM_SUBMIT_DIR
echo ${SCRATCH_DIR}	# tell user where scratch directory is
# make sure scrath directory exists
mkdir -p ${SCRATCH_DIR}execution_output/
mkdir -p ${SCRATCH_DIR}results/

# IF USING OPENBLAS
# module load ${BLAS_MODULE_DIR} # load libraries for openblas
# export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" # set the number of cores to be used

echo "python path: ${PYTHON3_PATH}"
echo "sampling script: ${SAMPLING_SCRIPT}"
echo "execution parameters ${EXECUTION_PARAMETERS}"
echo
# execute the job - 100% necessary for execution_parameters to be interpreted as a string
${PYTHON3_PATH} ${SAMPLING_SCRIPT} "${EXECUTION_PARAMETERS}" ${SCRATCH_DIR} ${SLURM_JOB_ID}

sleep 5
# retrive useful information about the job's execution and then print that info
EXECSTATS=$(ssh $SLURM_CLUSTER_NAME sacct -j $SLURM_JOBID --format=CPUTime,CPUTimeRAW,AveVMSize)

echo
echo "Execution stats: "
echo ${EXECSTATS}
echo "if there are no values it means the job didn't run long enough to record values"

# ls -al "${SCRATCH_DIR}execution_output/"
ls -al "${SCRATCH_DIR}results/"

echo mv --force "${COPY_FROM}"* "${COPY_TO}"
mv --force "${COPY_FROM}"* "${COPY_TO}"
# HACKS
# echo mv --force "${COPY_FROM}${SLURM_JOBID}"* "${COPY_TO}"
# mv --force "${COPY_FROM}${SLURM_JOBID}"* "${COPY_TO}"

# some user feedback
echo "Output file was here: $SLURMD_NODENAME:${SCRATCH_DIR}results/${SLURM_JOB_NAME}"
echo "Output files should be here now: ${ROOT_DIR}results/"

# finish up
cd "$SLURM_SUBMIT_DIR"
echo Job Script Finished Successfully
#
####