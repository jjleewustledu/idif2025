#!/bin/bash
# -*- coding: utf-8 -*-
# submit-mintun1984.sh
# ===================
#
# This script submits Mintun 1984 model analysis jobs to SLURM for processing PET data.
#
# The script configures and submits a SLURM job with specific resource requirements for analyzing
# CMRO2 using the Mintun 1984 model. It sets up email notifications, resource allocations, and 
# outputs job information to stdout/stderr.
#
# Structure:
# ---------
# 1. SLURM configuration
#    - Job name and output files
#    - Email notifications
#    - Resource requirements (CPUs, memory, time)
#    - Partition and account settings
# 2. Job information output
#    - Prints SLURM environment variables
#    - Logs execution details
#
# Usage:
# -----
# ./submit-mintun1984.sh <input_func> <v1_file> <ks_file>
#
# Arguments:
# ---------
# input_func : str
#     Path to the input function file (RadialArteryIO or BoxcarIO)
# v1_file : str
#     Path to the V1 parameter file
# ks_file : str
#     Path to the Ks parameter file
#
# Resources:
# ---------
# - 30 CPUs per task
# - 3GB memory per CPU
# - 48 hour time limit
# - Tier 2 CPU partition
#
# Notes:
# -----
# The script is configured for the Aristeidis Sotiras research group's
# reservation and account on the SLURM cluster.
#
# See Also:
# --------
# submit-radial-artery.sh : Script for radial artery input function analysis
# submit-boxcar.sh : Script for boxcar input function analysis
# submit-tissue.sh : Script for tissue analysis

# SLURM

## Job Name
#SBATCH -J submit_async

## Save output and error files in the following folder
# #SBATCH -o SLURM/output-submit-main-async.log
# #SBATCH -e SLURM/error-submit-main-async.log

## emailing
#SBATCH --mail-user=jjlee@wustl.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT

## resources
#SBATCH --priority=0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=3G
#SBATCH --time=48:00:00
#SBATCH --reservation=Aris_group
#SBATCH --account=aristeidis_sotiras
#SBATCH --partition=tier2_cpu

## send useful job information to stdout
echo "------------------------------------------------------"
echo "SLURM: sbatch is running on ${SLURM_SUBMIT_HOST}"
echo "SLURM: executing queue is ${SLURM_JOB_PARTITION}"
echo "SLURM: working directory is ${SLURM_SUBMIT_DIR}"
echo "SLURM: job identifier is ${SLURM_JOB_ID}"
echo "SLURM: job name is ${SLURM_JOB_NAME}"
echo "SLURM: cores per node ${SLURM_CPUS_ON_NODE}"
echo "SLURM: cores per task ${SLURM_CPUS_PER_TASK}"
echo "SLURM: node list ${SLURM_JOB_NODELIST}"
echo "SLURM: number of nodes for job ${SLURM_JOB_NUM_NODES}"
echo "------------------------------------------------------"

( echo -e "Executing in: \c"; pwd )
( echo -e "Executing at: \c"; date )

## send useful job information to stderr
echo "------------------------------------------------------" 1>&2
echo "SLURM: sbatch is running on ${SLURM_SUBMIT_HOST}" 1>&2
echo "SLURM: executing queue is ${SLURM_JOB_PARTITION}" 1>&2
echo "SLURM: working directory is ${SLURM_SUBMIT_DIR}" 1>&2
echo "SLURM: job identifier is ${SLURM_JOB_ID}" 1>&2
echo "SLURM: job name is ${SLURM_JOB_NAME}" 1>&2
echo "SLURM: cores per node ${SLURM_CPUS_ON_NODE}" 1>&2
echo "SLURM: cores per task ${SLURM_CPUS_PER_TASK}" 1>&2
echo "SLURM: node list ${SLURM_JOB_NODELIST}" 1>&2
echo "SLURM: number of nodes for job ${SLURM_JOB_NUM_NODES}" 1>&2
echo "------------------------------------------------------" 1>&2

( echo -e "Executing in: \c"; pwd ) 1>&2
( echo -e "Executing at: \c"; date ) 1>&2

# Project scripting

# See also PycharmProjects/dynesty/idif204.
#  1 parcel requires < 0.35 GB memory, 309 instances of multiprocessing.Pool, <1 h for 100 nlive, ~24 h for 1000 nlive.
#  For 100 nlive:  request nodes with 11 GB memory, 32 cores, 10 h.
#  For 1000 nlive:  request nodes with 11 GB memory, 32 cores, 240 h.

set -e
trap 'echo "Script submit_main_async.sh exited with error: $?" >&2; exit $?' ERR

the_main="Mintun1984Context.py"
inputf=$1
pet=$2
v1=$3
ks=$4
nlive=4000
filepath="${pet%/*}"
base="${pet##*/}"
fileprefix="${base%.*}"
fileprefix="${fileprefix%.*}"
date2save=$(date +"%m-%d-%y")

echo "Executing ${the_main}" 1>&2
echo "inputf is ${inputf}" 1>&2
echo "pet is ${pet}" 1>&2
echo "v1 is ${v1}" 1>&2
echo "ks is ${ks}" 1>&2
echo "nlive is ${nlive}" 1>&2

# MAIN Command

export CCHOME=/home/jjlee && \
export PATH="${CCHOME}/miniconda3/envs/dynesty12/bin:${PATH}" && \
python ${the_main} "${inputf}" "${pet}" "${v1}" "${ks}" "${nlive}" > "${filepath}/${fileprefix}-submit-${date2save}.log"
