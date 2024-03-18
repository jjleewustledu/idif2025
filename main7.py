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

from Raichle1983Model import Raichle1983Model
from Mintun1984Model import Mintun1984Model
from Huang1980ModelVenous import Huang1980ModelVenous


def the_tag(nlive: float, tag_rc="rc1p85", tag_vrc="vrc1"):
    """
    Generate a tag string based on specified parameters.

    :param nlive: The value of nlive.
    :type nlive: float
    :param tag_rc: The value of tag_rc. Default is "rc1p85".
    :type tag_rc: str
    :param tag_vrc: The value of tag_vrc. Default is "vrc1".
    :type tag_vrc: str
    :return: The generated tag string.
    :rtype: str
    """
    # __file__ gives the relative path of the script
    file_path = __file__
    file_name = os.path.basename(file_path)
    tag, _ = os.path.splitext(file_name)
    return f"{tag}-{tag_rc}-{tag_vrc}-{nlive}"


def work(tidx, data: dict):
    """
    :param tidx: The index of the time point at which to calculate the model.

    :param data: A dictionary of input data for the method.

    :return: A dictionary containing the results of the method.

    The 'work' method performs calculations based on the provided data dictionary.

    The method starts by printing the contents of the 'data' dictionary using the 'pprint' function.

    Next, it extracts the values of 'nlive' and 'venous_recovery_coefficient' from the 'data' dictionary and assigns them to variables '_nlive' and '_vrc' respectively.

    The method then calls the 'the_tag' function with the value of '_nlive' and 'tag_vrc' keyword argument set to 'vrc{_vrc}'. The returned value is assigned to the variable '_tag'.

    Depending on the value of 'pet_measurement' in the 'data' dictionary, the method creates an instance of one of three models:
    - If 'trc-ho' is in 'pet_measurement', it creates an instance of 'Raichle1983Model' and assigns it to the variable '_tcm'. The model is initialized with the values of 'input_function
    *', 'pet_measurement', 'truths', 'nlive', and 'tag'.
    - If 'trc-oo' is in 'pet_measurement', it creates an instance of 'Mintun1984Model' and assigns it to the variable '_tcm'. The model is initialized with the values of 'input_function
    *', 'pet_measurement', 'truths', 'nlive', and 'tag'.
    - If 'trc-fdg' is in 'pet_measurement', it creates an instance of 'Huang1980ModelVenous' and assigns it to the variable '_tcm'. The model is initialized with the values of 'input_function
    *', 'pet_measurement', 'truths', 'nlive', 'tag', and 'venous_recovery_coefficient'.

    If none of the above conditions are met, an empty dictionary is returned.

    The method then calls the 'run_nested_for_indexed_tac' method of the '_tcm' object, passing 'tidx' as the argument. The return value is assigned to the variable '_package'.

    The '_tcm' object is added to the '_package' dictionary using the key 'tcm'.

    Finally, the '_package' dictionary is returned as the result of the method.
    """

    pprint(data)
    _nlive = data["nlive"]
    _vrc = data["venous_recovery_coefficient"]
    _tag = the_tag(_nlive, tag_vrc=f"vrc{_vrc}")

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
        _tcm = Huang1980ModelVenous(
            data["input_function"],
            data["pet_measurement"],
            truths=[0.069, 0.003, 0.002, 0.000, 12.468, -9.492, 0.020],
            nlive=_nlive,
            tag=_tag,
            venous_recovery_coefficient=_vrc)
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
    martinv1s = []
    raichlekss = []

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
            _rho_pred, _, _, _ = tcm.signalmodel(tcm.data(_qm))
            rhos_pred.append(_rho_pred)
            resids.append(np.sum(_rho_pred - tcm.RHO) / np.sum(tcm.RHO))
            martinv1s.append(np.squeeze(tcm.martin_v1_measurement["img"]))
            raichlekss.append(np.squeeze(tcm.raichle_ks_measurement["img"]))
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
        "rho_pred": np.vstack(rhos_pred),
        "resid": np.array(resids),
        "martinv1": np.array(martinv1s),
        "raichleks": np.vstack(raichlekss)}

    packages[0]["tcm"].save_results(package1, tag=the_tag(Nlive, tag_vrc=f"vrc{vrc}"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
