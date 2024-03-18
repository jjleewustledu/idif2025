# This is a sample Python script.
import logging
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
from pprint import pprint
from multiprocessing import Pool
import logging

import numpy as np

from Boxcar import Boxcar
from RadialArtery import RadialArtery


def the_tag():
    """
    Returns the tag extracted from the filename of the script.

    :return: The tag extracted from the filename.
    :rtype: str
    """
    # __file__ gives the relative path of the script
    file_path = __file__
    file_name = os.path.basename(file_path)
    tag, _ = os.path.splitext(file_name)
    return tag


def work(tidx, data: dict):
    """
    Perform some work by Boxcar or RadialArtery with the given parameters.

    :param tidx: The tidx parameter.
    :param data: The data parameter.

    :return: The results of the work.
    """

    pprint(data)
    _nlive = data["nlive"]
    _tag = the_tag() + "-" + str(_nlive)

    if "MipIdif_idif" in data["input_function"]:
        _tcm = Boxcar(
            data["input_function"],
            truths=[
                13.8, 12.3, 53.9,
                0.668, 7.66, 1.96, -1.38, -0.023, 60.0,
                0.074, 0.027, 0.014,
                2.427,
                0.008],
            nlive=_nlive,
            tag=_tag)
    elif "TwiliteKit-do-make-input-func-nomodel_inputfunc" in data["input_function"]:
        _tcm = RadialArtery(
            data["input_function"],
            truths=[
                18.399, 6.656, .118,
                4.040, 3.924, 3.914, -2.322, -0.736, 28.411,
                0.319, 0.020, 0.040,
                2.552,
                0.026],
            nlive=_nlive,
            tag=_tag)
    else:
        return {}
        # raise RuntimeError(__name__ + ".work: data['pet_measurement'] -> " + data["pet_measurement"])

    _results = _tcm.run_nested()
    return _results


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # the_data objects for work

    # ============ SPOT TESTING ============
    # input_func_kind = "twil"
    # petdir = os.path.join(
    #     os.getenv("SINGULARITY_HOME"),
    #     "CCIR_01211", "derivatives", "sub-108293", "ses-20210421150523", "pet")
    # pet = os.path.join(
    #     petdir,
    #     "sub-108293_ses-20210421150523_trc-oo_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames_timeAppend-4"
    #     "-ParcSchaeffer-reshape-to-schaeffer-select-4.nii.gz")
    # Nparcels = 4
    # Nlive = 300

    input_func_kind = sys.argv[1]
    pet = sys.argv[2]
    try:
        Nparcels = int(sys.argv[3])
    except ValueError:
        sys.exit(1)
    try:
        Nlive = int(sys.argv[4])
    except ValueError:
        Nlive = 8000
    try:
        vrc = int(sys.argv[5])
    except ValueError:
        vrc = 1

    fqfp, _ = os.path.splitext(pet)
    fqfp, _ = os.path.splitext(fqfp)
    logging.basicConfig(
        filename=fqfp + ".log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s")

    trunc_idx = pet.find("proc-")
    prefix = pet[:trunc_idx]
    if "idif".lower() in input_func_kind.lower():
        if "fdg" in prefix:
            input_func = (prefix +
                          "proc-MipIdif_idif_dynesty-Boxcar-ideal-embed.nii.gz")
        else:
            input_func = (prefix +
                          "proc-MipIdif_idif_dynesty-Boxcar-ideal.nii.gz")
    elif ("twil".lower() in input_func_kind.lower() or
          "aif".lower() in input_func_kind.lower()):
        if "fdg" in prefix:
            input_func = (
                    prefix +
                    "proc-TwiliteKit-do-make-input-func-nomodel_inputfunc_dynesty-RadialArtery-ideal-embed.nii.gz")
        else:
            input_func = (
                    prefix +
                    "proc-TwiliteKit-do-make-input-func-nomodel_inputfunc_dynesty-RadialArtery-ideal.nii.gz")
    else:
        raise RuntimeError(__name__ + ": input_func_kind -> " + input_func_kind)

    the_data = {
        "input_function": input_func,
        "pet_measurement": pet,
        "nlive": Nlive,
        "venous_recovery_coefficient": vrc}
    tindices = list(range(Nparcels))  # parcs & segs in Nick Metcalf's Schaeffer parcellations

    # do multi-processing

    with Pool() as p:
        p.starmap(work, [(tidx, the_data) for tidx in tindices])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
