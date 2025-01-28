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
    pattern_if="trc-oo_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc-RadialArteryIO-ideal.nii.gz"
    pattern_v1="trc-co_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer-twilite_martinv1.nii.gz"
    pattern_ks="trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-schaeffer-TissueIO-RadialArtery-qm.nii.gz"
    ;;
  *)
    pattern_if="trc-oo_proc-MipIdif_idif-BoxcarIO-ideal.nii.gz"
    pattern_v1="trc-co_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer-idif_martinv1.nii.gz"
    pattern_ks="trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-schaeffer-TissueIO-Boxcar-qm.nii.gz"
    ;;
esac
echo "using pattern for input func:  ${pattern_if}"

# global variables
pattern_pet="trc-oo_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer.nii.gz"
submit_main="${HOME}/PycharmProjects/dynesty/idif2024/submit-mintun1984.sh"
tissue_context="${HOME}/PycharmProjects/dynesty/idif2024/Mintun1984Context.py"
derivatives="${SINGULARITY_HOME}/CCIR_01211/derivatives"
subs=("sub-108293" "sub-108237" "sub-108254" "sub-108250" "sub-108284" "sub-108306")

# Find v1 and ks files for a subject
find_v1_ks_files() {
    local session_path=$1
    local files_found=()
    local last_file=""

    # Find v1 file
    while IFS= read -r line; do
        files_found+=("$line")
    done < <(find "$session_path" -type f -name "*$pattern_v1*" | sort)

    # Get last v1 file
    if [ ${#files_found[@]} -gt 0 ]; then
        last_file="${files_found[-1]}"
    fi
    local v1_file="$last_file"
    
    # Reset for ks search
    files_found=()
    last_file=""

    # Find ks file 
    while IFS= read -r line; do
        files_found+=("$line")
    done < <(find "$session_path" -type f -name "*$pattern_ks*" | sort)

    # Get last ks file
    if [ ${#files_found[@]} -gt 0 ]; then
        last_file="${files_found[-1]}"
    fi
    local ks_file="$last_file"

    # Return v1 file and ks file
    printf '%s\n%s\n' "$v1_file" "$ks_file"
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
    local v1_file=$3
    local ks_file=$4

    echo "Submitting job with: ${submit_main} ${tissue_context} ${if_file} ${pet_file} ${v1_file} ${ks_file}"    
    ### sbatch "${submit_main}" "${tissue_context}" "${if_file}" "${pet_file}" "${v1_file}" "${ks_file}"
}

# Process each subject
for sub in "${subs[@]}"; do
    echo "Processing subject: $sub"
    sub_path="$derivatives/$sub"
    
    # First find v1 and ks files for this subject
    v1_file=""
    ks_file=""
    
    # Look through all sessions for v1 and ks files
    while IFS= read -r session; do
        readarray -t model_files < <(find_v1_ks_files "$session")
        if [ -n "${model_files[0]}" ] && [ -z "$v1_file" ]; then
            v1_file="${model_files[0]}"
        fi
        if [ -n "${model_files[1]}" ] && [ -z "$ks_file" ]; then
            ks_file="${model_files[1]}"
        fi
        # Break if we found both files
        if [ -n "$v1_file" ] && [ -n "$ks_file" ]; then
            break
        fi
    done < <(find "$sub_path" -maxdepth 1 -type d -name "ses-*")
    
    # Check if we found the required files
    if [ -z "$v1_file" ] || [ -z "$ks_file" ]; then
        echo "Warning: Could not find v1 file or ks file for subject $sub" >&2
        continue
    fi
    
    echo "Found v1 file: $v1_file"
    echo "Found ks file: $ks_file"
    
    # Now process each session with these v1 and ks files
    while IFS= read -r session; do
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
            submit_single_job "$if_file" "$pet_file" "$v1_file" "$ks_file"
        else
            ### echo "Warning: Could not find matching pair of files in session $session" >&2
        fi
    done < <(find "$sub_path" -maxdepth 1 -type d -name "ses-*")
done






