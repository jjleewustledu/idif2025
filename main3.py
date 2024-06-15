# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import time, sys, os, re
from Boxcar import Boxcar
from RadialArtery import RadialArtery
from multiprocessing import Pool

singularity = "/HOME/usr/jjlee/mnt/CHPC_scratch/Singularity"
template = [
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108293", "ses-20210421144815", "pet",
        "sub-108293_ses-20210421144815_trc-co_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108293", "ses-20210421150523", "pet",
        "sub-108293_ses-20210421150523_trc-oo_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108293", "ses-20210421152358", "pet",
        "sub-108293_ses-20210421152358_trc-ho_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108293", "ses-20210421154248", "pet",
        "sub-108293_ses-20210421154248_trc-oo_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
       "CCIR_01211", "derivatives", "sub-108293", "ses-20210421155709", "pet",
       "sub-108293_ses-20210421155709_trc-fdg_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108237", "ses-20221031100910", "pet",
        "sub-108237_ses-20221031100910_trc-co_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108237", "ses-20221031102320", "pet",
        "sub-108237_ses-20221031102320_trc-oo_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108237", "ses-20221031103712", "pet",
        "sub-108237_ses-20221031103712_trc-oo_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108237", "ses-20221031110638", "pet",
        "sub-108237_ses-20221031110638_trc-ho_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108237", "ses-20221031113804", "pet",
        "sub-108237_ses-20221031113804_trc-fdg_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108254", "ses-20221116095143", "pet",
        "sub-108254_ses-20221116095143_trc-co_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108254", "ses-20221116100858", "pet",
        "sub-108254_ses-20221116100858_trc-oo_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108254", "ses-20221116102328", "pet",
        "sub-108254_ses-20221116102328_trc-oo_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108254", "ses-20221116104751", "pet",
        "sub-108254_ses-20221116104751_trc-ho_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108254", "ses-20221116115244", "pet",
        "sub-108254_ses-20221116115244_trc-fdg_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108250", "ses-20221207093856", "pet",
        "sub-108250_ses-20221207093856_trc-co_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108250", "ses-20221207095507", "pet",
        "sub-108250_ses-20221207095507_trc-oo_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108250", "ses-20221207100946", "pet",
        "sub-108250_ses-20221207100946_trc-oo_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108250", "ses-20221207102944", "pet",
        "sub-108250_ses-20221207102944_trc-ho_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108250", "ses-20221207104909", "pet",
        "sub-108250_ses-20221207104909_trc-fdg_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108284", "ses-20230220093702", "pet",
        "sub-108284_ses-20230220093702_trc-co_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108284", "ses-20230220095210", "pet",
        "sub-108284_ses-20230220095210_trc-oo_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108284", "ses-20230220101103", "pet",
        "sub-108284_ses-20230220101103_trc-oo_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108284", "ses-20230220103226", "pet",
        "sub-108284_ses-20230220103226_trc-ho_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108284", "ses-20230220112328", "pet",
        "sub-108284_ses-20230220112328_trc-ho_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108306", "ses-20230227103048", "pet",
        "sub-108306_ses-20230227103048_trc-co_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108306", "ses-20230227104631", "pet",
        "sub-108306_ses-20230227104631_trc-oo_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108306", "ses-20230227112148", "pet",
        "sub-108306_ses-20230227112148_trc-oo_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108306", "ses-20230227113853", "pet",
        "sub-108306_ses-20230227113853_trc-ho_proc-procspecifier.nii.gz"),
    os.path.join(singularity,
        "CCIR_01211", "derivatives", "sub-108306", "ses-20230227115809", "pet",
        "sub-108306_ses-20230227115809_trc-fdg_proc-procspecifier.nii.gz")
       ]

pet = [s.replace("derivatives", "sourcedata") for s in template]
pet = [s.replace("procspecifier", "delay0-BrainMoCo2-createNiftiMovingAvgFrames") for s in pet]
idf = [s.replace("procspecifier", "MipIdif_idif") for s in template]
idf_mdl = [s.replace("procspecifier", "MipIdif_idif_dynesty-Boxcar-ideal") for s in template]
aif = [s.replace("procspecifier", "TwiliteKit-do-make-input-func-nomodel_inputfunc") for s in template]
aif_mdl = [s.replace("procspecifier", "TwiliteKit-do-make-input-func-nomodel_inputfunc_dynesty-RadialArtery-ideal") for s in template]

def work(idx):
    # Use a breakpoint in the code line below to debug your script.

    _ifm = template[idx]
    _km = km[idx]
    print(f'working on: {_ifm}, {_km}')  # Press ⌘F8 to toggle the breakpoint.

    if "TwiliteKit" in _ifm and _km:
        ra = RadialArtery(_ifm, _km, nlive=1000)  # 1000
        ra.run_nested()
    if "MipIdif" in _ifm:
        bc = Boxcar(_ifm, nlive=1000)  # 1000
        bc.run_nested()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = range(30)
    with Pool() as p:
        p.map(work, data)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
