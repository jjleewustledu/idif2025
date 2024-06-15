#!/bin/bash

set -e
trap 'echo "Script submit_all_tz3108.sh exited with error: $?" >&2; exit $?' ERR

derivatives="${SINGULARITY_HOME}/TZ3108/derivatives"
subs=("sub-bud" "sub-cheech" "sub-lou" "sub-ollie" "sub-stan" "sub-wuzzy")
if_type="aif"
model_types=("SpectralAnalysis")  ###("Ichise2002VascModel" "Huang1980Model")
proc="sub-*_trc-tz3108_*-tacs.nii.gz"
submit_main="${HOME}/PycharmProjects/dynesty/idif2024/submit_main_tz3108.sh"

for sub in "${subs[@]}"; do
  containing_folder="$derivatives/$sub"
  files=()

  # use a while-read loop to feed the find results into the array
  while IFS= read -r line; do
      files+=("$line")
  done < <(find "$containing_folder" -type f -name "$proc")

  for afile in "${files[@]}"; do
    for mdl_type in "${model_types[@]}"; do
      echo "sbatch \"${submit_main}\" \"${if_type}\" \"${afile}\" \"${mdl_type}\""
      sbatch "${submit_main}" "${if_type}" "${afile}" "${mdl_type}"
    done
  done
done
