# This is a sample Python script.
from __future__ import absolute_import
import logging
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
from pprint import pprint
from multiprocessing import Pool
import logging

import numpy as np

from Raichle1983Model import Raichle1983Model
from Mintun1984Model import Mintun1984Model
from Huang1980Model import Huang1980Model
from six.moves import range


def the_tag():
    # __file__ gives the relative path of the script
    file_path = __file__
    file_name = os.path.basename(file_path)
    tag, _ = os.path.splitext(file_name)
    return tag


def work(tidx, data: dict):
    """"""

    pprint(data)
    _nlive = data["nlive"]
    _tag = the_tag() + "-" + str(_nlive)

    if "trc-ho" in data["pet_measurement"]:
        _tcm = Raichle1983Model(
            data["input_function"],
            data["pet_measurement"],
            truths=[0.014, 0.866, 0.013, 17.6, -8.6, 0.049],
            nlive=_nlive,
            tag=_tag)
    elif "trc-oo" in data["pet_measurement"]:
        _tcm = Mintun1984Model(
            data["input_function"],
            data["pet_measurement"],
            truths=[0.511, 0.245, 0.775, 5, -5, 0.029],
            nlive=_nlive,
            tag=_tag)
    elif "trc-fdg" in data["pet_measurement"]:
        _tcm = Huang1980Model(
            data["input_function"],
            data["pet_measurement"],
            truths=[0.069, 0.003, 0.002, 0.000, 12.468, -9.492, 0.020],
            nlive=_nlive,
            tag=_tag)
    else:
        return {}
        # raise RuntimeError(__name__ + ".work: data['pet_measurement'] -> " + data["pet_measurement"])

    _package = _tcm.run_nested_for_indexed_tac(tidx)
    _package["tcm"] = _tcm
    return _package


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
        Nlive = 300

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
        "nlive": Nlive}
    tindices = list(range(Nparcels))  # parcs & segs in Nick Metcalf's Schaeffer parcellations

    # do multi-processing

    with Pool() as p:
        packages = p.starmap(work, [(tidx, the_data) for tidx in tindices])

    # re-order and save packages

    ress = []
    logzs = []
    informations = []
    qms = []
    qls = []
    qhs = []
    martinv1s = []
    raichlekss = []
    rhos_pred = []
    resids = []

    for package in packages:
        try:
            tcm = package["tcm"]
            ress.append(package["res"])
            rd = package["res"].asdict()
            logzs.append(rd["logz"][-1])
            informations.append(rd["information"][-1])
            _qm, _ql, _qh = tcm.solver.quantile(package["res"])
            qms.append(_qm)
            qls.append(_ql)
            qhs.append(_qh)
            martinv1s.append(tcm.martin_v1_measurement)
            raichlekss.append(tcm.raichle_ks_measurement)
            _rho_pred, _, _, _ = tcm.signalmodel(tcm.data(_qm))
            rhos_pred.append(_rho_pred)
            resids.append(np.sum(_rho_pred - tcm.rho) / np.sum(tcm.rho))
        except Exception as e:
            # catch any error to enable graceful exit with writing whatever results were incompletely obtained
            logging.exception(__name__ + ": error in tcm -> " + str(e), exc_info=True)

    package1 = {
        "res": ress,
        "logz": np.array(logzs),
        "information": np.array(informations),
        "qm": np.vstack(qms),
        "ql": np.vstack(qls),
        "qh": np.vstack(qhs),
        "martinv1": np.vstack(martinv1s),
        "raichleks": np.vstack(raichlekss),
        "rho_pred": np.vstack(rhos_pred),
        "resid": np.array(resids)}

    packages[0]["tcm"].results_save(package1, tag=the_tag() + "-rc1p85-" + str(Nlive))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
