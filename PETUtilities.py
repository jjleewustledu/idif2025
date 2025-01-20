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

from __future__ import absolute_import
from copy import deepcopy
import os
from pathlib import Path
import re

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


class PETUtilities:

    @staticmethod
    def data2times(data: dict) -> np.ndarray:
        """Retrieve times array from data dictionary."""
        if "times" in data:
            return data["times"]
        timesMid = data["timesMid"]
        taus = data["taus"]
        return timesMid - taus / 2

    @staticmethod
    def data2taus(data: dict) -> np.ndarray:
        """Retrieve taus array from data dictionary."""
        if "taus" in data:
            return data["taus"] 
        timesMid = data["timesMid"]
        times = data["times"]
        return 2 * (timesMid - times)

    @staticmethod
    def data2timesMid(data: dict) -> np.ndarray:
        """Retrieve timesMid array from data dictionary."""
        if "timesMid" in data:
            return data["timesMid"]
        times = data["times"]
        taus = data["taus"]
        return times + taus / 2
    
    @staticmethod
    def data2timesInterp(data: dict) -> np.ndarray:
        """Retrieve times array from data dictionary and interpolate to 1-second resolution from 0 sec."""
        tinterp0 = data["times"][0]  # sec
        tinterpF = data["times"][-1] + data["taus"][-1] - 1  # sec  
        N_tinterp = (tinterpF - tinterp0 + 1).astype(int)  # N of 1-sec samples
        return np.linspace(tinterp0, tinterpF, N_tinterp)  # e.g., [0.1, 1.1, 2.1, ..., N_tinterp+0.1]
    
    @staticmethod
    def data2timesMidInterp(data: dict) -> np.ndarray:
        """Retrieve timesMid array from data dictionary and interpolate to 1-second resolution from 0.5 sec."""
        return PETUtilities.data2timesInterp(data) + 0.5  # sec
    
    @staticmethod
    def decay_correct(tac: dict) -> dict:
        _tac = deepcopy(tac)
        img = _tac["img"] * np.power(2, _tac["timesMid"] / _tac["halflife"])
        _tac["img"] = img
        return _tac

    @staticmethod
    def decay_uncorrect(tac: dict) -> dict:
        _tac = deepcopy(tac)
        img = _tac["img"] * np.power(2, -_tac["timesMid"] / _tac["halflife"])
        _tac["img"] = img
        return _tac

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
        pth, fp = PETUtilities.fileparts(fqfn)
        return os.path.join(pth, fp)

    @staticmethod
    def interpdata(data: dict, ref: dict, kind: str="linear") -> dict:
        """ Interpolates data dictionary to 1-second resolution specified by ref dictionary."""
        data1 = deepcopy(data)
        assert "img" in data1, "data must have 'img' entry"
        assert "times" in data1, "data must have 'times' entry"
        assert "timesMid" in data1, "data must have 'timesMid' entry"
        assert "taus" in data1, "data must have 'taus' entry"
        assert "timesMid" in ref, "ref must have 'timesMid' entry"

        times1 = PETUtilities.data2timesInterp(ref)
        timesMid1 = PETUtilities.data2timesMidInterp(ref)
        data1["img"] = PETUtilities.interpimg(timesMid1, data["timesMid"], data["img"], kind=kind)
        data1["times"] = times1
        data1["timesMid"] = timesMid1
        data1["taus"] = 2 * (timesMid1 - times1)
        return data1
    
    @staticmethod
    def interpimg(timesMid1: NDArray, timesMid: NDArray, img: NDArray, kind: str="linear") -> NDArray:
        """ Interpolates img, corresponding to timesMid, to timesMidNew, squeezing everything. 
            timesMidNew and timesMid may be 1D, [1, N_time], or [N_time, 1]. 
            img may be 1D, [1, N_time], [N_time, 1], or [N_pos, N_time]. """        
        if img.squeeze().ndim > 1 or not kind == "linear":
            f = interp1d(timesMid.squeeze(), img.squeeze(), kind=kind, axis=1)
            return f(timesMid1.squeeze())
        else:
            return np.interp(timesMid1.squeeze(), timesMid.squeeze(), img.squeeze())

    @staticmethod
    def parse_branching_ratio(fileprefix: str) -> float:
        """ http://www.turkupetcentre.net/petanalysis/branching_ratio.html """
        iso = PETUtilities.parse_isotope(fileprefix)
        if iso == "11C":
            return 0.998
        if iso == "13N":
            return 0.998
        if iso == "15O":
            return 0.999
        if iso == "18F":
            return 0.967
        if iso == "68Ga" or iso == "68Ge":
            return 0.891
        if iso == "64Cu":
            return 0.1752
        if iso == "82Rb":
            return 0.950
        if iso == "89Zr":
            return 0.22
        if iso == "124I":
            return 0.26        
        raise ValueError(f"branching ratio not identifiable from fileprefix {fileprefix}")   
        
    @staticmethod
    def parse_halflife(fileprefix: str) -> float:
        """ http://www.turkupetcentre.net/petanalysis/decay.html """
        iso = PETUtilities.parse_isotope(fileprefix)
        if iso == "11C":
            return 20.340253 * 60  # sec * sec/min
        if iso == "13N":
            return 9.97 * 60  # sec * sec/min
        if iso == "15O":
            return 122.2416  # sec
        if iso == "18F":
            return 109.771 * 60  # sec * sec/h
        if iso == "22Na":
            return 2.602 * 365.25 * 86400  # +/-0.15 y * day/y * sec/day
        if iso == "62Cu":
            return 9.67 * 60  # min * sec/min
        if iso == "64Cu":
            return 12.7003 * 3600  # h * sec/h
        if iso == "68Ga":
            return 67.719 * 60  # min * sec/min
        if iso == "68Ge":
            return 270.8 * 86400  # days * sec/day
        if iso == "82Rb":
            return 75  # sec
        if iso == "89Zr":
            return 3.27 * 86400  # days * sec/day
        if iso == "124I":
            return 4.176 * 86400  # days * sec/day
        if iso == "137Cs":
            return 30.17 * 365.25 * 86400  # +/-9.5 y * day/y * sec/day
        raise ValueError(f"halflife not identifiable from fileprefix {fileprefix}")

    @staticmethod
    def parse_isotope(filename: str) -> str:
        basename = Path(filename).stem 
        trc_11c = ["trc-cglc", "trc-cs1p1", "trc-mdl", "trc-nmsp", "trc-pib", "trc-raclopride"]
        trc_15o = ["trc-co", "trc-oc", "trc-oo", "trc-ho"]
        trc_18f = [
            "trc-asem", "trc-av45", "trc-av1451", "trc-azan", "trc-fdg", 
            "trc-florbetaben", "trc-florbetapir", "trc-flortaucipir", "trc-flutemetamol", 
            "trc-gtp1", "trc-jnj-64326067", "trc-mk6240", "trc-pi2620", "trc-ro948", 
            "trc-tz3108", "trc-vat"]

        if any(x in basename for x in trc_11c):
            return "11C"
        if any(x in basename for x in trc_15o):
            return "15O"
        if any(x in basename for x in trc_18f):
            return "18F"
        
        # likely reasonable to guess 18F
        warnings.warn(f"isotope not identifiable from fileprefix {filename}; guessing 18F", RuntimeWarning)
        return "18F"

    @staticmethod
    def parse_tracer(filename: str) -> str:
        """Extract tracer name from (f.q.) filename between 'trc-' and '_'."""
        basename = Path(filename).stem 
        match = re.search(r'trc-([^_]+)', basename)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"tracer not identifiable from filename {fqfn}")    

    @staticmethod
    def slice_parc(img: NDArray, xindex: int) -> NDArray:
        """ slices img that is N_parc x N_time by the xindex for the parcel of interest,
            returns a 1D array of length N_time. """
        
        assert img.ndim <= 2, "img must be 1D or 2D"
        return img[xindex].copy()
    
    @staticmethod
    def slide(rho: NDArray, t: NDArray, dt: float, halflife: float=None) -> NDArray:
        """ slides rho by dt seconds, optionally decays it by halflife. """

        if abs(dt) < 0.1:
            return rho
        rho = PETUtilities.interpimg(t - dt, t, rho)  # copy of rho array
        if halflife:
            rho = rho * np.power(2, -dt / halflife)
        return rho.copy()
        