# The MIT License (MIT)
#
# Copyright (c) 2024 - Present: John J. Lee.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
from typing import Any
from IOInterface import IOInterface
import os
import json
import numpy as np
import nibabel as nib
import pandas as pd
import sys
import inspect
from pathlib import Path
from copy import deepcopy

from PETUtilities import PETUtilities
from DynestyContext import DynestyContext


class BaseIO(IOInterface):
    """Base implementation of IOInterface with common functionality."""

    @property
    def fqfp(self):
        return None

    @property
    def results_fqfp(self):
        return None

    @staticmethod
    def data2times(data: dict) -> np.ndarray:
        return PETUtilities.data2times(data)

    @staticmethod
    def data2taus(data: dict) -> np.ndarray:
        return PETUtilities.data2taus(data)

    @staticmethod
    def data2timesMid(data: dict) -> np.ndarray:
       return PETUtilities.data2timesMid(data)
    
    @staticmethod
    def data2timesInterp(data: dict) -> np.ndarray:
        return PETUtilities.dadata2timesInterpta2tinterp(data)

    @staticmethod
    def fileparts(fqfn: str) -> tuple:
        """
        Extracts full path and basename without any extensions from a filepath.        
        Args:
            fqfn: String containing fully qualified filename            
        Returns:
            tuple containing:
            - parent_path: Full directory path
            - basename: Filename without any extensions            
        Example:
            "/path/to/file.nii.gz" -> ("/path/to", "file")
        """
        path = Path(fqfn)
        parent_path = str(path.parent)
        basename = path.stem  # Removes last extension
        
        # Handle multiple extensions (e.g., .nii.gz)
        while "." in basename:
            basename = Path(basename).stem
            
        return parent_path, basename 
    
    @staticmethod
    def fqfileprefix(fqfn: str) -> str:
        """Extract fully qualified fileprefix from fully qualified filename."""
        pth, fp = BaseIO.fileparts(fqfn)
        return os.path.join(pth, fp)

    def load_nii(self, fqfn: str) -> dict:
        """Load a NIfTI file and associated json."""
        if not fqfn.endswith(".nii.gz"):
            fqfn += ".nii.gz"
        if not os.path.isfile(fqfn):
            raise FileNotFoundError(f"{fqfn} was not found")

        # defer to nibabel
        nii = nib.load(fqfn)

        # load json
        jfile = self.fqfileprefix(fqfn) + ".json"
        with open(jfile) as f:
            j = json.load(f)

        # assemble dict from nibabel and json
        fqfp = self.fqfileprefix(fqfn)
        niid = {
            "fqfp": fqfp,
            "nii": nii,
            "img": nii.get_fdata().squeeze(),
            "json": j,
            "halflife": PETUtilities.parse_halflife(fqfp)}
        if "timesMid" in j:
            niid["timesMid"] = np.array(j["timesMid"], dtype=float).squeeze()
        if "taus" in j:
            niid["taus"] = np.array(j["taus"], dtype=float).squeeze()
        if "times" in j:
            niid["times"] = np.array(j["times"], dtype=float).squeeze()
        if "martinv1" in j:
            niid["martinv1"] = np.array(j["martinv1"], dtype=float).squeeze()
        if "raichleks" in j:
            niid["raichleks"] = np.array(j["raichleks"], dtype=float).squeeze()

        return self.trim_nii_dict(niid)
            
    def load_pickled(self, fqfn: str) -> Any:
        """Load data from a pickle file."""
        if not fqfn.endswith(".pickle"):
            fqfn += ".pickle"            
        if not os.path.isfile(fqfn):
            raise FileNotFoundError(f"{fqfn} was not found")
        with open(fqfn, "rb") as f:
            return pickle.load(f)
        
    def save_csv(self, data: dict, fqfn: str = None) -> None:
        """Save data to a CSV file."""
        if not fqfn:
            raise ValueError("fqfn must be a valid filename")
        if not fqfn.endswith(".csv"):
            fqfn += ".csv"
        df = pd.DataFrame(data)
        df.to_csv(fqfn)

    def save_nii(self, data: dict, fqfn: str | None = None) -> None:
        """Save data to a NIfTI file and associated json."""
        if not fqfn:
            raise ValueError("fqfn must be a valid filename")
        if not fqfn.endswith(".nii.gz"):
            fqfn += ".nii.gz"
        try:
            # defer to nibabel
            _data = deepcopy(data)
            nii = _data["nii"]
            nii = nib.Nifti1Image(_data["img"], nii.affine, nii.header)
            nib.save(nii, fqfn)

            # save updated json
            jfile1 = self.fqfileprefix(fqfn) + ".json"
            with open(jfile1, "w") as f:
                json.dump(data["json"], f, indent=4)
        except Exception as e:
            print(f"{self.__class__.__name__}.save_nii: caught Exception {e}, but proceeding", file=sys.stderr)
            print(f"{fqfn} may be missing or malformed")

    def save_pickled(self, data: Any, fqfn: str | None = None) -> None:
        """Save by pickling."""        
        if not fqfn:
            raise ValueError("fqfn must be a valid filename")
        if not fqfn.endswith(".pickle"):
            fqfn += ".pickle"        
        with open(fqfn, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    @staticmethod
    def trim_nii_dict(niid: dict, time_last: float = None) -> dict:
        """ Examines niid and trims copies of its entries to
            (i) remove inviable temporal samples indicated by np.isnan(timesMid)
            (ii) remove temporal samples occurring after time_last
            (iii) trim all niid entries that appear to be temporal samples. """

        if not isinstance(niid, dict):
            raise TypeError(f"Expected niid to be dict but it has type {type(niid)}.")

        img = niid["img"].copy()
        timesMid = niid["timesMid"].copy()
        taus = niid["taus"].copy()
        times = niid["times"].copy()

        # nans are not viable; also adjust viability with time_last
        viable = ~np.isnan(timesMid)        
        if time_last is not None:
            viable = viable * (timesMid <= time_last)

        if img.ndim == 1 and len(img) == len(timesMid):
            niid.update({
                "img": img[viable], 
                "timesMid": timesMid[viable], 
                "taus": taus[viable],
                "times": times[viable]
            })
        elif img.ndim == 2 and img.shape[1] == len(timesMid):
            niid.update({
                "img": img[:, viable], 
                "timesMid": timesMid[viable], 
                "taus": taus[viable],
                "times": times[viable]
            })
        return niid


class RadialArteryIO(BaseIO):
    """I/O operations specific to RadialArtery models."""
    
    def __init__(self, context: DynestyContext):
        self.context = context
    
    @property
    def fqfp(self):
        return self.context.data.input_func_measurement["fqfp"]

    @property
    def results_fqfp(self):
        return self.fqfp + "-" + self.__class__.__name__

    def load_kernel(self, fqfn: str) -> dict:
        """Load kernel measurement data."""
        if not fqfn.endswith(".nii.gz"):
            fqfn += ".nii.gz"
        if not os.path.isfile(fqfn):
            raise FileNotFoundError(f"{fqfn} was not found")
        nii = nib.load(fqfn)
        img = nii.get_fdata()
        return {
            "fqfp": self.fqfileprefix(fqfn),
            "img": np.array(img, dtype=float).squeeze()
        }


class TrivialArteryIO(BaseIO):
    """I/O operations specific to TrivialArtery models."""
    pass


class BoxcarIO(BaseIO):
    """I/O operations specific to Boxcar models."""
    
    def __init__(self, context: DynestyContext):
        self.context = context
    
    @property
    def fqfp(self):
        return self.context.data.input_func_measurement["fqfp"]

    @property
    def results_fqfp(self):
        return self.fqfp + "-" + self.__class__.__name__


class TissueIO(BaseIO):
    """I/O operations specific to TissueModel models."""

    def __init__(self, context: DynestyContext):
        self.context = context

    @property
    def fqfp(self):
        return self.context.data.tissue_measurement["fqfp"]

    @property
    def results_fqfp(self):
        fqfp1 = (
            self.fqfp + "-" + 
            self.__class__.__name__ + "-" + 
            self.context.ARTERY.__class__.__name__ + "-" + 
            self.context.data.tag)
        fqfp1 = fqfp1.replace("ParcSchaeffer-reshape-to-schaeffer-", "")
        fqfp1 = fqfp1.replace("ModelAndArtery", "")
        fqfp1 = fqfp1.replace("Model", "")
        fqfp1 = fqfp1.replace("Radial", "")
        return fqfp1
