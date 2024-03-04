#!/bin/bash

set -e
trap 'echo "Script submit_all.sh exited with error: $?" >&2; exit $?' ERR

derivatives="${SINGULARITY_HOME}/CCIR_01211/derivatives"
subs=("sub-108293" "sub-108237" "sub-108254" "sub-108250" "sub-108284" "sub-108306")
if_types=("twil" "idif")
proc="proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames*-ParcSchaeffer-reshape-to-schaeffer-schaeffer.nii.gz"
submit_main="${HOME}/PycharmProjects/dynesty/idif2024/submit_main_async.sh"
Nparcels=309

for sub in "${subs[@]}"; do
  containing_folder="$derivatives/$sub"
  files=()

  # use a while-read loop to feed the find results into the array
  while IFS= read -r line; do
      files+=("$line")
  done < <(find "$containing_folder" -type f -name "*$proc*")

  for afile in "${files[@]}"; do
    for type in "${if_types[@]}"; do
      sbatch "${submit_main}" "${type}" "${afile}" ${Nparcels}
    done
  done
done
