#!/bin/bash

set -e
trap 'echo "Script submit_inputf.sh exited with error: $?" >&2; exit $?' ERR

# determine the pattern for input func
case "$1" in
  -h|--help)
    echo "Usage: $0 [artery|twilite]"
    echo "  Providing 'artery' or 'twilite' will use RadialArteryIO-ideal.nii.gz" 
    echo "  Otherwise BoxcarIO-ideal.nii.gz will be used"
    exit 0
    ;;
  artery|twilite)
    pattern_if="trc-ho_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc-RadialArteryIO-ideal.nii.gz"
    ;;
  *)
    pattern_if="trc-ho_proc-MipIdif_idif-BoxcarIO-ideal.nii.gz"
    ;;
esac
echo "using pattern for input func:  ${pattern_if}"

# global variables
pattern_pet="trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer.nii.gz"
submit_main="${HOME}/PycharmProjects/dynesty/idif2024/submit-tissue.sh"
tissue_context="${HOME}/PycharmProjects/dynesty/idif2024/Raichle1983Context.py"
derivatives="${SINGULARITY_HOME}/CCIR_01211/derivatives"
subs=("sub-108293" "sub-108237" "sub-108254" "sub-108250" "sub-108284" "sub-108306")
len_subs="${#subs[@]}"

# Find all session folders for subjects
find_sessions() {
  local sessions=()
  
  # Loop through subjects and find sessions
  for sub in "${subs[@]}"; do
    # Use while-read loop to collect session folders into array
    while IFS= read -r line; do
      sessions+=("$line")
    done < <(find "$derivatives/$sub" -maxdepth 1 -type d -name "ses-*")
  done

  # Return the array
  printf '%s\n' "${sessions[@]}"
}

# Find matching files in a single session path
find_files_in_session() {
    local session_path=$1
    local files_found=()

    # Find pet file
    while IFS= read -r line; do
        files_found+=("$line")
    done < <(find "$session_path" -type f -name "*$pattern_pet*")

    if [ ${#files_found[@]} -gt 1 ]; then
        echo "Error: Multiple PET files found in $session_path matching pattern: $pattern_pet" >&2
        exit 1
    fi

    # Store pet file
    local pet_file="${files_found[0]}"
    files_found=()

    # Find input function file
    while IFS= read -r line; do
        files_found+=("$line")
    done < <(find "$session_path" -type f -name "*$pattern_if*")

    if [ ${#files_found[@]} -gt 1 ]; then
        echo "Error: Multiple input function files found in $session_path matching pattern: $pattern_if" >&2
        exit 1
    fi

    # Store input function file
    local if_file="${files_found[0]}"

    # Return array with input function file first, then pet file
    printf '%s\n%s\n' "$if_file" "$pet_file"
}

# Submit a single job to SLURM with input function and PET files
submit_single_job() {
    local if_file=$1
    local pet_file=$2

    echo "Submitting job with: ${submit_main} ${tissue_context} ${if_file} ${pet_file}"    
    sbatch "${submit_main}" "${tissue_context}" "${if_file}" "${pet_file}"
}

sessions=($(find_sessions))

# Process each session
for session in "${sessions[@]}"; do
    echo "Processing session: $session"
    
    # Get matching files for this session
    readarray -t files < <(find_files_in_session "$session")
    
    # Check if we found both files
    if [ ${#files[@]} -eq 2 ]; then
        if_file="${files[0]}"
        pet_file="${files[1]}"
        
        # Skip if either file is empty
        if [ -z "$if_file" ] || [ -z "$pet_file" ]; then
            echo "Warning: Missing files in session $session" >&2
            continue
        fi
        
        # Submit job for this pair
        submit_single_job "$if_file" "$pet_file"
    else
        echo "Warning: Could not find matching pair of files in session $session" >&2
    fi
done






