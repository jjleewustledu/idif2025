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


from multiprocessing import Pool
import dynesty
from dynesty import utils as dyutils
import numpy as np
from numba import jit
from numpy.typing import NDArray

from TissueSolver import TissueSolver


@jit(nopython=True)
def prior_transform(
    u: np.ndarray,
    sigma: float
) -> np.ndarray:
    v = u
    v[0] = u[0] * 0.016 + 0.0011  # f (1/s)
    v[1] = u[1] * 1.95 + 0.05  # \lambda (cm^3/mL)
    v[2] = u[2] * 0.0272 + 0.0011  # ps (mL cm^{-3}s^{-1})
    v[3] = u[3] * 40 - 10  # t_0 (s)
    v[4] = u[4] * sigma  # sigma ~ fraction of A0
    return v


@jit(nopython=True)
def loglike(
    v: np.ndarray,
    rho: np.ndarray,
    timesMid: np.ndarray,
    taus: np.ndarray,
    rho_input_func_interp: np.ndarray,
    isidif: bool
) -> float:
    rho_pred, _, _, _ = signalmodel(v, timesMid, taus, rho_input_func_interp, isidif)
    sigma = v[-1]
    residsq = (rho_pred - rho) ** 2 / sigma ** 2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma ** 2))
    if not np.isfinite(loglike):
        loglike = -1e300
    return loglike


@jit(nopython=True)
def signalmodel(
    v: np.ndarray,
    timesMid: np.ndarray,
    taus: np.ndarray,
    rho_input_func_interp: np.ndarray,
    isidif: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Raichle1983Model is only valid for [15O] """
    HL = 122.2416  # [15O]
    ALPHA = np.log(2) / HL

    f = v[0]
    lamb = v[1]
    ps = v[2]
    t_0 = v[3]

    n_times = rho_input_func_interp.shape[0]
    timesIdeal = np.arange(0, n_times)

    # Find indices where input function exceeds 5% of max
    indices = np.where(rho_input_func_interp > 0.05 * np.max(rho_input_func_interp))        
    # Handle case where no values exceed threshold
    if len(indices[0]) == 0:
        idx_a = 1  # Default to 1 if no values exceed threshold
    else:
        idx_a = max(indices[0][0], 1)  # Take first index but ensure >= 1

    # slide input function to fit
                    
    if not isidif:
        # slide input function to left, 
        # since its measurements is delayed by radial artery cannulation
        rho_input_func_interp = slide(
            rho_input_func_interp, 
            timesIdeal, 
            -timesIdeal[idx_a], 
            0) 
    rho_input_func_interp = slide(
        rho_input_func_interp, 
        timesIdeal, 
        t_0, 
        HL)

    # propagate input function

    m = 1 - np.exp(-ps / f)
    propagator = np.exp(-m * f * timesIdeal / lamb - ALPHA * timesIdeal)

    # numpy ------------------------------------------------------------
    # conv_h2o = np.convolve(propagator, artery_h2o, mode="full")
    # conv_o2 = np.convolve(propagator, artery_o2, mode="full")        
    # numba jit --------------------------------------------------------
    n_propagator = len(propagator)
    n_artery = len(rho_input_func_interp) 
    n_conv = n_propagator + n_artery - 1        
    conv_h2o = np.zeros(n_conv)     
    for i in range(n_conv):
        for j in range(max(0, i - n_propagator + 1), min(i + 1, n_artery)):
            conv_h2o[i] += propagator[i - j] * rho_input_func_interp[j]
    rho1 = m * f * conv_h2o

    # package compartments

    rho_ideal = rho1[:n_times]  # rho_ideal is interpolated to the input function times
    if not isidif:
        rho_pred = np.interp(timesMid, timesIdeal, rho_ideal)
    else:
        rho_pred = apply_boxcar(rho_ideal, timesMid, taus)
    return rho_pred, timesMid, rho_ideal, timesIdeal


@jit(nopython=True)
def apply_boxcar(rho: np.ndarray, timesMid: np.ndarray, taus: np.ndarray) -> np.ndarray:
    times0_int = (timesMid - taus / 2).astype(np.int_)
    timesF_int = (timesMid + taus / 2).astype(np.int_)

    # Original implementation with loop ---------------------------------------
    # rho_sampled = np.full(times0_int.shape, np.nan)
    # for idx, (t0, tF) in enumerate(zip(times0_int, timesF_int)):
    #     rho_sampled[idx] = np.mean(rho[t0:tF])        
    # return np.nan_to_num(rho_sampled, 0)
    # Optimized implementation using cumsum ------------------------------------
    # padding rho with 0 at beginning 
    cumsum = np.cumsum(np.concatenate((np.zeros(1), rho)))
    rho_sampled = (cumsum[timesF_int] - cumsum[times0_int]) / taus
    return np.nan_to_num(rho_sampled, 0)


@jit(nopython=True)
def slide(rho: np.ndarray, t: np.ndarray, dt: float, halflife: float = None) -> np.ndarray:
    """ slides rho by dt seconds, optionally decays it by halflife. """

    if abs(dt) < 0.1:
        return rho
    rho = np.interp(t - dt, t, rho)  # copy of rho array
    if halflife:
        return rho * np.power(2, -dt / halflife)
    else:
        return rho


class Raichle1983Solver(TissueSolver):
    """Solver implementing the Raichle 1983 tissue model for PET data analysis.

    This class implements the tissue model described in Raichle et al. 1983 [1]_ for analyzing
    PET data using dynamic nested sampling. The model accounts for blood flow (f),
    blood-tissue partition coefficient (λ), permeability-surface area product (ps),
    time offset (t0), and arterial dispersion (τa).

    Args:
        context: Context object containing PET data and configuration.

    Attributes:
        context: Reference to the context object.
        data: Reference to the context's data object containing PET measurements.

    Example:
        >>> context = TissueContext(data_dict)
        >>> solver = Raichle1983Solver(context)
        >>> results = solver.run_nested()
        >>> qm, ql, qh = solver.quantile(results)

    References:
        .. [1] Raichle ME, Martin WR, Herscovitch P, Mintun MA, Markham J.
               Brain blood flow measured with intravenous H2(15)O. II. Implementation and validation.
               J Nucl Med. 1983 Sep;24(9):790-8. PMID: 6604140.
    """
    def __init__(self, context):
        super().__init__(context)

    @property
    def labels(self):
        return [r"$f$", r"$\lambda$", r"ps", r"$t_0$", r"$\sigma$"]

    @staticmethod
    def _loglike(selected_data: dict):
        # Cache values using nonlocal for faster access
        rho = selected_data["rho"]
        timesMid = selected_data["timesMid"]
        taus = selected_data["taus"]
        rho_input_func_interp = selected_data["rho_input_func_interp"]
        isidif = selected_data["isidif"]

        # Check dimensions
        if rho.ndim != 1:
            raise ValueError("rho must be 1-dimensional")
        if rho_input_func_interp.ndim != 1:
            raise ValueError("rho_input_func_interp must be 1-dimensional") 
        
        # Create wrapper that matches dynesty's expected signature
        def wrapped_loglike(v):
            nonlocal rho, timesMid, taus, rho_input_func_interp, isidif
            return loglike(
                v,
                rho,
                timesMid,
                taus, 
                rho_input_func_interp,
                isidif)
        return wrapped_loglike

    @staticmethod
    def _prior_transform(selected_data: dict):
        # Create wrapper that matches dynesty's expected signature
        sigma = selected_data["sigma"]

        def wrapped_prior_transform(v):
            nonlocal sigma
            return prior_transform(v, sigma)
        return wrapped_prior_transform
    
    @staticmethod
    def _run_nested(selected_data: dict) -> dyutils.Results:
        if selected_data["resume"]:
            sampler = dynesty.DynamicNestedSampler.restore(selected_data["checkpoint_file"])
        else:
            loglike = Raichle1983Solver._loglike(selected_data)
            prior_transform = Raichle1983Solver._prior_transform(selected_data)
            sampler = dynesty.DynamicNestedSampler(
                loglikelihood=loglike,
                prior_transform=prior_transform,
                ndim=selected_data["ndim"],
                sample=selected_data["sample"],
                nlive=selected_data["nlive"],
                rstate=selected_data["rstate"]
            )
        sampler.run_nested(
            checkpoint_file=selected_data["checkpoint_file"], 
            print_progress=selected_data["print_progress"], 
            resume=selected_data["resume"],
            wt_kwargs={"pfrac": selected_data["pfrac"]}
        )
        return sampler.results
        
    def _run_nested_pool(
            self,
            checkpoint_file: list[str] | None = None,
            print_progress: bool = False,
            resume: bool = False,
            parc_index: list[int] | tuple[int, ...] | NDArray | None = None
    ) -> list[dyutils.Results]:
    
        if not parc_index:
            parc_index = range(len(self.data.rho))
        elif isinstance(parc_index, np.ndarray):
            parc_index = parc_index.tolist()
        
        if checkpoint_file and len(checkpoint_file) != len(parc_index):
            raise ValueError("checkpoint_file must be a list of strings matching length of parc_index")

        # Create args list suitable for Pool.starmap()
        args = []
        for i, pidx in enumerate(parc_index):
            cf = checkpoint_file[i] if isinstance(checkpoint_file, list) else None
            selected_data = {
                "rho": self.data.rho[pidx],
                "timesMid": self.data.timesMid,
                "taus": self.data.taus,
                "rho_input_func_interp": self.data.rho_input_func_interp,
                "isidif": self.data.isidif,
                "sigma": self.data.sigma,
                "ndim": self.ndim,
                "sample": self.data.sample,
                "nlive": self.data.nlive,
                "rstate": self.data.rstate,
                "checkpoint_file": cf,
                "resume": resume,
                "pfrac": self.data.pfrac,
                "print_progress": False
            }
            args.append(selected_data)
        
        # Sequential execution for testing
        # _results = [Raichle1983Solver.__run_nested(*arg) for arg in args]
        # self._set_cached_dynesty_results(_results)
        # return _results    

        # Use multiprocessing Pool to parallelize execution is incompatible with instance methods
        with Pool() as p:
            _results = p.starmap(Raichle1983Solver._run_nested, [(arg,) for arg in args])
            self._set_cached_dynesty_results(_results)
        return _results
        
    def _run_nested_single(
            self,
            checkpoint_file: str | None = None,
            print_progress: bool = False,
            resume: bool = False,
            parc_index: int | None = None
    ) -> dyutils.Results:      
        args = {
            "rho": self.data.rho[parc_index],
            "timesMid": self.data.timesMid,
            "taus": self.data.taus,
            "rho_input_func_interp": self.data.rho_input_func_interp,
            "isidif": self.data.isidif,
            "sigma": self.data.sigma,
            "ndim": self.ndim,
            "sample": self.data.sample,
            "nlive": self.data.nlive,
            "rstate": self.data.rstate,
            "checkpoint_file": checkpoint_file,
            "resume": resume,
            "pfrac": self.data.pfrac,
            "print_progress": print_progress
        }

        _results = Raichle1983Solver._run_nested(args)
        self._set_cached_dynesty_results(_results)
        return _results
    
    def signalmodel(
            self,
            v: list | tuple | NDArray,
            parc_index: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ parc_index supports the API"""

        v = np.array(v, dtype=float)
        if not isinstance(v, np.ndarray) or v.ndim != 1 or len(v) != self.ndim:
            raise ValueError(f"v must be 1-dimensional array of length {self.ndim}")
        return signalmodel(
            v, 
            self.data.timesMid,
            self.data.taus,
            self.data.rho_input_func_interp,
            self.data.isidif
        )
    
    def loglike(
            self,
            v: list | tuple | NDArray,
            parc_index: int | None = None
    ) -> float:
        """ parc_index supports the API"""
        
        v = np.array(v, dtype=float)
        if not isinstance(v, np.ndarray) or v.ndim != 1 or len(v) != self.ndim:
            raise ValueError(f"v must be 1-dimensional array of length {self.ndim}")
        return loglike(
            v,
            self.data.rho,
            self.data.timesMid,
            self.data.taus,
            self.data.rho_input_func_interp,
            self.data.isidif
        )
