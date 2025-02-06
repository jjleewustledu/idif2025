#!/bin/bash
# -*- coding: utf-8 -*-
# loop-submit-inputf.sh
# ====================
#
# This script submits input function analysis jobs to SLURM for multiple subjects and sessions.
#
# The script processes PET data to generate both radial artery and boxcar input functions
# by submitting jobs using submit-radial-artery.sh and submit-boxcar.sh.
#
# Structure:
# ---------
# 1. Global variable definitions for paths and subjects
# 2. Helper function for job submission
# 3. Main execution for both input function types
#
# Functions:
# ---------
# submit_jobs(proc, submit_main)
#     Submits SLURM jobs for input function processing
#     
#     Parameters:
#     ----------
#     proc : str
#         Pattern to match input function files
#     submit_main : str
#         Path to submission script to use
#
# Usage:
# -----
# ./loop-submit-inputf.sh
#
# Notes:
# -----
# The script submits two types of jobs:
# - Radial artery input function generation using submit-radial-artery.sh
# - Boxcar input function generation using submit-boxcar.sh
#
# The script expects specific file naming patterns:
# - *proc-TwiliteKit-do-make-input-func-nomodel_inputfunc.nii.gz for radial artery
# - *proc-MipIdif_idif.nii.gz for boxcar

set -e
trap 'echo "Script submit_inputf.sh exited with error: $?" >&2; exit $?' ERR

derivatives="${SINGULARITY_HOME}/CCIR_01211/derivatives"
subs=("sub-108293" "sub-108237" "sub-108254" "sub-108250" "sub-108284" "sub-108306")
len_subs="${#subs[@]}"

submit_jobs() {
  local proc=$1
  local submit_main=$2

  for ((i=0; i<$len_subs; i++)); do
    sub="${subs[$i]}"
    containing_folder="$derivatives/$sub"
    files=()

    # use a while-read loop to feed the find results into the array
    while IFS= read -r line; do
        files+=("$line")
    done < <(find "$containing_folder" -type f -name "*$proc*")

    for afile in "${files[@]}"; do
      echo "sbatch \"${submit_main}\" \"${afile}\""
      sbatch "${submit_main}" "${afile}"
    done
  done
}

# submit_radial_artery.sh
proc="proc-TwiliteKit-do-make-input-func-nomodel_inputfunc.nii.gz"
submit_main="${HOME}/PycharmProjects/dynesty/idif2025/submit-radial-artery.sh"
submit_jobs "$proc" "$submit_main"

# submit_boxcar.sh
proc="proc-MipIdif_idif.nii.gz"
submit_main="${HOME}/PycharmProjects/dynesty/idif2025/submit-boxcar.sh"
submit_jobs "$proc" "$submit_main"
