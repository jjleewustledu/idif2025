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
from pathlib import Path
import re

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


class PETData:

    def __init__(self, fqfn: str):
        self._fqfn = fqfn

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
    def interpimg(timesNew: NDArray, times: NDArray, img: NDArray, kind: str="linear") -> NDArray:
        """ Interpolates img, corresponding to times, to timesNew, squeezing everything. 
            timesNew and times may be 1D, [1, N_time], or [N_time, 1]. 
            img may be 1D, [1, N_time], [N_time, 1], or [N_pos, N_time]. """
        
        if img.squeeze().ndim > 1 or not kind == "linear":
            f = interp1d(times.squeeze(), img.squeeze(), kind=kind, axis=1)
            return f(timesNew.squeeze())
        else:
            return np.interp(timesNew.squeeze(), times.squeeze(), img.squeeze())

    @staticmethod
    def parse_branching_ratio(fileprefix: str) -> float:
        """ http://www.turkupetcentre.net/petanalysis/branching_ratio.html """
        iso = PETData.parse_isotope(fileprefix)
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
        iso = PETData.parse_isotope(fileprefix)
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
    def parse_isotope(fileprefix: str) -> str:
        trc_11c = ["trc-cglc", "trc-cs1p1", "trc-mdl", "trc-nmsp", "trc-pib", "trc-raclopride"]
        trc_15o = ["trc-co", "trc-oc", "trc-oo", "trc-ho"]
        trc_18f = [
            "trc-asem", "trc-av45", "trc-av1451", "trc-azan", "trc-fdg", 
            "trc-florbetaben", "trc-florbetapir", "trc-flortaucipir", "trc-flutemetamol", 
            "trc-gtp1", "trc-jnj-64326067", "trc-mk6240", "trc-pi2620", "trc-ro948", 
            "trc-tz3108", "trc-vat"]

        if any(x in fileprefix for x in trc_11c):
            return "11C"
        if any(x in fileprefix for x in trc_15o):
            return "15O"
        if any(x in fileprefix for x in trc_18f):
            return "18F"
        
        # likely reasonable to guess 18F
        warnings.warn(f"isotope not identifiable from fileprefix {fileprefix}; guessing 18F", RuntimeWarning)
        return "18F"

    @staticmethod
    def parse_tracer(fqfn: str) -> str:
        """Extract tracer name from (f.q.) filename between 'trc-' and '_'."""
        basename = Path(fqfn).stem 
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
        rho = PETData.interpimg(t - dt, t, rho)  # copy of rho array
        if halflife:
            return rho * np.power(2, -dt / halflife)
        else:
            return rho
        