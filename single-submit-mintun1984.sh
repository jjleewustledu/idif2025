#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 [108284|108306] [-h|--help]"
    exit 0
fi

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      echo "Usage: $0 108284|108306"
      echo "  Providing one of the supported irregular subject ids will launch submit-mintun1984.sh for it."
      echo "  Otherwise do nothing."
      exit 0
      ;;
    108284)
      sub=sub-108284
      ses=ses-20230220101103
      sesco=ses-20230220093702
      sesho=ses-20230220112328
      ;;
    108306)
      sub=sub-108306
      ses=ses-20230227112148
      sesco=ses-20230227103048
      sesho=ses-20230227113853	
      ;;
  esac
done

submit_main=${HOME}/PycharmProjects/dynesty/idif2025/submit-mintun1984.sh

derivatives=/home/jjlee/Singularity/CCIR_01211/derivatives
inputf=${derivatives}/${sub}/${ses}/pet/${sub}_${ses}_trc-oo_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc-RadialArteryIO-ideal.nii.gz
pet=${derivatives}/${sub}/${ses}/pet/${sub}_${ses}_trc-oo_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames_timeAppend-4-ParcSchaeffer-reshape-to-schaeffer-schaeffer.nii.gz
v1=${derivatives}/${sub}/${sesco}/pet/${sub}_${sesco}_trc-co_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer-idif_martinv1.nii.gz
ks=${derivatives}/${sub}/${sesho}/pet/${sub}_${sesho}_trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-schaeffer-TissueIO-Artery-qm.nii.gz

sbatch "${submit_main}" "${inputf}" "${pet}" "${v1}" "${ks}"
