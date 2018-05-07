#!/bin/bash
# submit script for running memory profiling jobs on nlogn
# 
# 
# 
# 
# These variables are available to the job where you append PBS_O_ to the front
# HOME, LANG, LOGNAME, PATH, MAIL, SHELL, and TZ
#
################################ SETUP AND USER INFO ################################ 
#
INPUT_DIR="${ROOT_DIR}parameters/"
SCRATCH_DIR="/scratch/$USER"
MEMORY_PROFILER="/home/ngraymon/dev/ubuntu/16.04/bin/mprof"
PYTHON3="/home/ngraymon/dev/ubuntu/16.04/bin/python3"
SAMPLING_SCRIPT="$PBS_O_WORKDIR/engine.py"

# make sure the scratch directory is available
mkdir -p "/scratch/$USER"

echo "Running script on $HOSTNAME"
echo

echo "${SAMPLES} samples"
echo "${BLOCKS} block size"
echo "${BEADS} beads"
echo "${SURFACES} surfaces "
echo "${TEMP} temperature"
echo "${INPUT_DIR} input_dir"
echo "${MODES} modes"
echo "${NUMBER_OF_CORES} cores reserved"
echo "${MEMORY_RESERVED} GB of memory reserved"

echo "$PBS_JOBNAME"
echo "$PBS_JOBID"

#
################################ Meat & Potatoes ################################
# 

${MEMORY_PROFILER} run ${PYTHON3} ${SAMPLING_SCRIPT} ${SAMPLES} ${BLOCKS} ${BEADS} ${MODES} ${SURFACES} ${TEMP} ${INPUT_DIR} "${SCRATCH_DIR}/" ${PBS_JOBNAME}
echo "Output file was here: $HOSTNAME:${SCRATCH_DIR}/${PBS_JOBNAME}_denom.npy"

mv --force "${SCRATCH_DIR}/${PBS_JOBNAME}_denom.npy" "${ROOT_DIR}results/"
mv --force "${SCRATCH_DIR}/${PBS_JOBNAME}_numer.npy" "${ROOT_DIR}results/"
mv --force "${SCRATCH_DIR}/${PBS_JOBNAME}_expval" 	 "${ROOT_DIR}results/"

echo "Output files should be here now: ${ROOT_DIR}results/"

cd "$PBS_O_WORKDIR"
echo Job Script Finished Successfully 

#
#
#
#
####