# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
from pprint import pprint
from multiprocessing import Pool
import numpy as np

from Raichle1983ModelAndArtery import Raichle1983ModelAndArtery
from Mintun1984ModelAndArtery import Mintun1984ModelAndArtery
from Huang1980ModelAndArtery import Huang1980ModelAndArtery


def work(tidx, data: dict):
    """"""

    pprint(data)

    if "trc-ho" in data["pet_measurement"]:
        _tcm = Raichle1983ModelAndArtery(
            data["input_function"],
            data["pet_measurement"],
            truths=[
                13.8, 12.3, 53.9,
                0.668, 7.66, 1.96, -1.38, -0.023, 60.0,
                0.074, 0.027, 0.014,
                2.427,
                0.008,
                0.014, 0.866, 0.013, 17.6, -8.6, 0.049],
            nlive=100)
    elif "trc-oo" in data["pet_measurement"]:
        _tcm = Mintun1984ModelAndArtery(
            data["input_function"],
            data["pet_measurement"],
            truths=[
               13.8, 12.3, 53.9,
               0.668, 7.66, 1.96, -1.38, -0.023, 60.0,
               0.074, 0.027, 0.014,
               2.427,
               0.008,
               0.511, 0.245, 0.775, 5, -5, 0.029],
            nlive=100)
    elif "trc-fdg" in data["pet_measurement"]:
        _tcm = Huang1980ModelAndArtery(
            data["input_function"],
            data["pet_measurement"],
            truths=[
                13.2, 20.8, 59.5, 0.272, 6.25, 2.56, -1.21, -0.654, 11.7, 0.0678, 0.0466, 0.0389, 2.44, 0.0222,
                0.069, 0.003, 0.002, 0.000, 12.468, -9.492, 0.020],
            nlive=100)
    else:
        raise RuntimeError(__name__ + ".work: data['pet_measurement'] -> " + data["pet_measurement"])

    _package = _tcm.run_nested_for_indexed_tac(tidx)
    _package["tcm"] = _tcm
    return _package


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # data objects for work

    # petdir = os.path.join(
    #     os.getenv("SINGULARITY_HOME"),
    #     "CCIR_01211", "derivatives", "sub-108293", "ses-20210421150523", "pet")
    # idif = os.path.join(petdir, "sub-108293_ses-20210421150523_trc-oo_proc-MipIdif_idif.nii.gz")
    # pet = os.path.join(
    #     petdir,
    #     "sub-108293_ses-20210421150523_trc-oo_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-"
    #     "ParcSchaeffer-reshape-to-schaeffer-schaeffer.nii.gz")

    input_func_kind = sys.argv[1]
    pet = sys.argv[2]

    trunc_idx = pet.find("proc-")
    prefix = pet[:trunc_idx]
    if "idif".lower() in input_func_kind.lower():
        input_func = prefix + "proc-MipIdif_idif.nii.gz"
    elif ("twil".lower() in input_func_kind.lower() or
          "aif".lower() in input_func_kind.lower()):
        input_func = prefix + "proc-TwiliteKit-do-make-input-func-nomodel-recalibrated_inputfunc.nii.gz"
    else:
        raise RuntimeError(__name__ + ": input_func_kind -> " + input_func_kind)

    data = {
        "input_function": input_func,
        "pet_measurement": pet}
    tindices = list(range(309))  # parcs & segs in Nick Metcalf's Schaeffer parcellations

    # do multi-processing

    with Pool() as p:
        packages = p.starmap(work, [(tidx, data) for tidx in tindices])

    # re-order and save packages

    ress = []
    logzs = []
    informations = []
    qms = []
    qls = []
    qhs = []
    rhos_pred = []
    resids = []

    for package in packages:
        tcm = package["tcm"]
        ress.append(package["res"])
        rd = package["res"].asdict()
        logzs.append(rd["logz"][-1])
        informations.append(rd["information"][-1])
        _qm, _ql, _qh = tcm.solver.quantile(package["res"])
        qms.append(_qm)
        qls.append(_ql)
        qhs.append(_qh)
        _rho_pred, _, _, _ = tcm.signalmodel(tcm.data(_qm))
        rhos_pred.append(_rho_pred)
        resids.append(np.sum(_rho_pred - tcm.RHO) / np.sum(tcm.RHO))

    package1 = {
        "res": ress,
        "logz": np.array(logzs),
        "information": np.array(informations),
        "qm": np.vstack(qms),
        "ql": np.vstack(qls),
        "qh": np.vstack(qhs),
        "rho_pred": np.vstack(rhos_pred),
        "resid": np.array(resids)}

    packages[0]["tcm"].save_results(package1, tag="main5")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
