#!/bin/bash

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
submit_main="${HOME}/PycharmProjects/dynesty/idif2024/submit-radial-artery.sh"
submit_jobs "$proc" "$submit_main"

# submit_boxcar.sh
proc="proc-MipIdif_idif.nii.gz"
submit_main="${HOME}/PycharmProjects/dynesty/idif2024/submit-boxcar.sh"
submit_jobs "$proc" "$submit_main"
