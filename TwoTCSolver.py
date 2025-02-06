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
    v[0] = u[0] * 1e3 + 1  # k_1/k_2
    v[1] = u[1] * 0.5 + 0.00001  # k_2 (1/s)
    v[2] = u[2] * 1e3 + 1  # k_3/k_4
    v[3] = u[3] * 0.05 + 0.00001  # k_4 (1/s)
    v[4] = u[4] * 20  # t_0 (s)
    v[5] = u[5] * sigma  # sigma ~ fraction of A0
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
    assert rho.ndim == 1, "rho must be 1-dimensional"
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
    """ TwoTCSolver assumes all input params to be decay-corrected """

    K1 = v[0] * v[1]  # k1/k2 * k2
    k2 = v[1]
    k3 = v[2] * v[3]  # k3/k4 * k4
    k4 = v[3]
    t_0 = v[4]

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
            -timesIdeal[idx_a])
    rho_input_func_interp = slide(
        rho_input_func_interp,
        timesIdeal,
        t_0)

    # propagate input function

    k234 = k2 + k3 + k4
    bminusa = np.sqrt(np.power(k234, 2) - 4 * k2 * k4)
    alpha = 0.5 * (k234 - bminusa)
    beta = 0.5 * (k234 + bminusa)
    propagator_a = np.exp(-alpha * timesIdeal)
    propagator_b = np.exp(-beta * timesIdeal)

    # numpy ------------------------------------------------------------
    # conv_h2o = np.convolve(propagator, artery_h2o, mode="full")
    # conv_o2 = np.convolve(propagator, artery_o2, mode="full")
    # numba jit --------------------------------------------------------
    n_propagator = len(propagator_a)
    n_artery = len(rho_input_func_interp)
    n_conv = n_propagator + n_artery - 1
    conv_a = np.zeros(n_conv)
    for i in range(n_conv):
        for j in range(max(0, i - n_propagator + 1), min(i + 1, n_artery)):
            conv_a[i] += propagator_a[i - j] * rho_input_func_interp[j]
    conv_b = np.zeros(n_conv)
    for i in range(n_conv):
        for j in range(max(0, i - n_propagator + 1), min(i + 1, n_artery)):
            conv_b[i] += propagator_b[i - j] * rho_input_func_interp[j]
    conv_a = conv_a[:n_times]  # interpolated to the input function times
    conv_b = conv_b[:n_times]  # interpolated to the input function times

    # package compartments

    conv_2 = (k4 - alpha) * conv_a + (beta - k4) * conv_b
    conv_3 = conv_a - conv_b
    q2 = (K1 / bminusa) * conv_2
    q3 = (k3 * K1 / bminusa) * conv_3

    rho_ideal = rho_input_func_interp + q2 + q3
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
def slide(rho: np.ndarray, t: np.ndarray, dt: float, halflife: float = 0) -> np.ndarray:
    """ slides rho by dt seconds, optionally decays it by halflife. """

    if abs(dt) < 0.1:
        return rho
    rho = np.interp(t - dt, t, rho)  # copy of rho array
    if halflife > 0:
        return rho * np.power(2, -dt / halflife)
    else:
        return rho


class TwoTCSolver(TissueSolver):
    """Solver implementing the 2-tissue compartment model for PET data analysis.

    This class implements the tissue model described in Huang et al. 1980 [1] for analyzing
    PET data using dynamic nested sampling. The model accounts for blood flow (f),
    blood-tissue partition coefficient (λ), permeability-surface area product (ps),
    time offset (t0), and arterial dispersion (τa).  It uses K1 ~ v1 * k1.

    Args:
        context: Context object containing PET data and configuration.

    Attributes:
        context: Reference to the context object.
        data: Reference to the context's data object containing PET measurements.

    Example:
        >>> context = TissueContext(data_dict)
        >>> solver = Huang1980Solver(context)
        >>> results = solver.run_nested()
        >>> qm, ql, qh = solver.quantile(results)

    References:
        .. [1] Huang SC, Phelps ME, Hoffman EJ, Sideris K, Selin CJ, Kuhl DE.
               Noninvasive determination of local cerebral metabolic rate of glucose in man.
               Am J Physiol. 1980;238(1):E69-82. doi:10.1152/ajpendo.1980.238.1.E69
    """

    def __init__(self, context):
        super().__init__(context)

    @property
    def labels(self):
        return [r"$K_1/k_2$", r"$k_2$", r"$k_3/k_4$", r"$k_4$", r"$t_0$", r"$\sigma$"]    
    
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
            loglike = TwoTCSolver._loglike(selected_data)
            prior_transform = TwoTCSolver._prior_transform(selected_data)
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
        # _results = [Huang1980Solver.__run_nested(*arg) for arg in args]
        # self._set_cached_dynesty_results(_results)
        # return _results

        # Use multiprocessing Pool to parallelize execution is incompatible with instance methods
        with Pool() as p:
            _results = p.starmap(TwoTCSolver._run_nested, [(arg,) for arg in args])
            self._set_cached_dynesty_results(_results)
        return _results

    def _run_nested_single(
            self,
            checkpoint_file: str | None = None,
            print_progress: bool = False,
            resume: bool = False,
            parc_index: int | None = None
    ) -> dyutils.Results:
        if parc_index is None:
            raise ValueError("parc_index must be provided")
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

        _results = TwoTCSolver._run_nested(args)
        self._set_cached_dynesty_results(_results)
        return _results

    def signalmodel(
            self,
            v: list | tuple | NDArray,
            parc_index: int | None = None  # Unused parameter for API consistency
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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
            parc_index: int | None = None  # Unused parameter for API consistency
    ) -> float:

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
