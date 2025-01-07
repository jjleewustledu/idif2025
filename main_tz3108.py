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
from TZ3108 import TZ3108
from six.moves import range


def the_tag(nlive: float, tag_model: str):
    """    """

    # __file__ gives the relative path of the script
    file_path = __file__
    file_name = os.path.basename(file_path)
    a_tag, _ = os.path.splitext(file_name)
    return f"{a_tag}"


def work(tidx, data: dict):
    """    """

    pprint(data)
    _nlive = data["nlive"]
    _model = data["model"]
    _tag = the_tag(_nlive, _model)

    if "trc-tz3108" in data["pet_measurement"]:
        _tcm = TZ3108(
            data["input_function"],
            data["pet_measurement"],
            truths=[0.0023, 0.016, 0.00069, 0.0034, 0.012, 0.00012, 0.0074, 0.020],
            nlive=_nlive,
            tag=_tag,
            model=_model,
            delta_time=4,
            M=3)
    else:
        return {}
        # raise RuntimeError(__name__ + ".work: data['pet_measurement'] -> " + data["pet_measurement"])

    _package = _tcm.run_nested_for_indexed_tac(tidx)
    _package["tcm"] = _tcm
    return _package


def is_stackable(arrays):
    if not isinstance(arrays, (list, tuple)):
        return False
    if not all(isinstance(arr, np.ndarray) for arr in arrays):
        return False
    if len({arr.shape[1:] for arr in arrays}) != 1:
        return False
    return True


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # the_data objects for work

    input_func_kind = sys.argv[1]  # ignored, but maintain interface of TCM
    pet = sys.argv[2]
    try:
        Nparcels = int(sys.argv[3])
    except ValueError:
        Nparcels = 21
    try:
        Nlive = int(sys.argv[4])
    except ValueError:
        Nlive = 1000
    try:
        model = sys.argv[5]
    except ValueError:
        model = "LineModel"

    fqfp, _ = os.path.splitext(pet)
    fqfp, _ = os.path.splitext(fqfp)
    logging.basicConfig(
        filename=fqfp + ".log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s")

    trunc_idx = pet.find("-tacs")
    prefix = pet[:trunc_idx]
    input_func = (prefix + "-aif.nii.gz")

    the_data = {
        "input_function": input_func,
        "pet_measurement": pet,
        "nlive": Nlive,
        "model": model}
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
    rhos_pred = []
    resids = []

    for pidx, package in enumerate(packages):
        try:
            # save each package separately since this try-except block often has errors
            tag = f"{the_tag(Nlive, tag_model=model)}_{__name__}_pidx{pidx}"
            package["tcm"].pickle_results(package, tag=tag)

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
            resids.append(np.sum(_rho_pred - tcm.rho) / np.sum(tcm.rho))
        except Exception as e:
            # catch any error to enable graceful exit with writing whatever results were incompletely obtained
            logging.exception(__name__ + ": error in tcm -> " + str(e), exc_info=True)

    if is_stackable(qms):
        qms = np.vstack(qms)
    if is_stackable(qls):
        qls = np.vstack(qls)
    if is_stackable(qhs):
        qhs = np.vstack(qhs)
    if is_stackable(rhos_pred):
        rhos_pred = np.vstack(rhos_pred)

    package1 = {
        "res": ress,
        "logz": np.array(logzs),
        "information": np.array(informations),
        "qm": qms,
        "ql": qls,
        "qh": qhs,
        "rho_pred": rhos_pred,
        "resid": np.array(resids)}

    packages[0]["tcm"].save_results(package1, tag=the_tag(Nlive, tag_model=model))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
