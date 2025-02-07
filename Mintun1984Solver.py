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
    v[0] = u[0] * 0.9 + 0.1  # OEF
    v[1] = u[1] * 1.5 + 0.5  # frac. water of metab. at 90 s
    v[2] = u[2] * 0.95 + 0.05  # {v_{post} + 0.5 v_{cap}} / v_1
    v[3] = u[3] * 40 - 10  # t_0 (s)
    v[4] = u[4] * 29 + 1  # \tau_d (s)
    v[5] = u[5] * sigma  # sigma ~ fraction of A0
    return v


@jit(nopython=True)
def loglike(
    v: np.ndarray,
    rho: np.ndarray,
    timesMid: np.ndarray,
    taus: np.ndarray,
    rho_input_func_interp: np.ndarray,
    ks: np.ndarray,
    v1: float,
    isidif: bool
) -> float:
    assert rho.ndim == 1, "rho must be 1-dimensional"
    rho_pred, _, _, _ = signalmodel(v, timesMid, taus, rho_input_func_interp, ks, v1, isidif)
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
    ks: np.ndarray,
    v1: float,
    isidif: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Mintun1984Model is only valid for [15O] """
    HL = 122.2416  # [15O]
    ALPHA = np.log(2) / HL
    DENSITY_PLASMA = 1.03
    DENSITY_BLOOD = 1.06

    oef = v[0]  # extraction fraction
    f_h2o = v[1] * DENSITY_PLASMA / DENSITY_BLOOD  # fraction of water of metabolism at 90 s
    v_post_cap = v[2]  # volume of post-capillary and capillaryspace
    t_0 = v[3]  # delay of input function
    tau_dispersion = v[4]  # dispersion of input function
    f = ks[0]  # flow, s^{-1}
    lamb = ks[1]  # partition coefficient
    PS = ks[2]  # permeability surface area product, s^{-1}

    n_times = rho_input_func_interp.shape[0]
    timesIdeal = np.arange(0, n_times)

    # Find indices where input function exceeds 5% of max
    indices = np.where(rho_input_func_interp > 0.05 * np.max(rho_input_func_interp))
    # Handle case where no values exceed threshold
    if len(indices[0]) == 0:
        idx_a = 1  # Default to 1 if no values exceed threshold
    else:
        idx_a = max(indices[0][0], 1)  # Take first index but ensure >= 1

    # disperse input function, compatible with numba

    dispersion = np.exp(-timesIdeal**2 / (2 * tau_dispersion**2))
    z_dispersion = np.sum(dispersion)  # partition function of dispersion

    # numpy ------------------------------------------------------------
    # conv_result = np.convolve(rho_input_func_interp, dispersion, mode="full")
    # numba jit --------------------------------------------------------
    n_input = len(rho_input_func_interp)  # numba jit
    n_disp = len(dispersion)
    n_conv = n_input + n_disp - 1
    conv_result = np.zeros(n_conv)
    for i in range(n_conv):
        for j in range(max(0, i - n_disp + 1), min(i + 1, n_input)):
            conv_result[i] += rho_input_func_interp[j] * dispersion[i - j]
    rho_input_func_interp = conv_result[:n_times] / z_dispersion

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

    # estimate shape of water of metabolism

    # Find indices where input function exceeds 5% of max
    # indices = np.where(rho_input_func_interp > 0.05 * np.max(rho_input_func_interp))
    # Handle case where no values exceed threshold
    # if len(indices[0]) == 0:
    #     idx0 = 1  # Default to 1 if no values exceed threshold
    # else:
    #     idx0 = max(indices[0][0], 1)  # Take first index but ensure >= 1
    idx0 = 1
    idxU = min([idx0 + 90, n_times - 1])  # time of eval of magnitude of water of metab; cf. Mintun1984
    shape = np.zeros(n_times)
    n_times_1 = n_times - idx0 + 1
    if idxU == idx0:
        y = 1
    else:
        y = (n_times - idx0) / (idxU - idx0)
    shape[-n_times_1:] = np.linspace(0, y, n_times_1)  # shape(idxU) == 1
    timesDuc = np.zeros(n_times)
    timesDuc[idx0:] = np.linspace(0, n_times_1 - 2, n_times_1 - 1)
    shape_duc = shape * np.power(2, -(timesDuc - idxU + 1) / HL)  # decay-uncorrected

    # set scale of artery_h2o
    # activity of water of metab \approx activity of oxygen after 90 sec

    artery_h2o = f_h2o * rho_input_func_interp[idxU] * shape_duc

    # compartment 2, using m, f, lamb, compatible with numba

    artery_o2 = rho_input_func_interp - artery_h2o
    artery_o2[artery_o2 < 0] = 0

    m = 1 - np.exp(-PS / f)
    propagator = np.exp(-m * f * timesIdeal / lamb - ALPHA * timesIdeal)

    # numpy ------------------------------------------------------------
    # conv_h2o = np.convolve(propagator, artery_h2o, mode="full")
    # conv_o2 = np.convolve(propagator, artery_o2, mode="full")
    # numba jit --------------------------------------------------------
    n_propagator = len(propagator)
    n_artery = len(artery_h2o)
    n_conv = n_propagator + n_artery - 1
    conv_h2o = np.zeros(n_conv)
    conv_o2 = np.zeros(n_conv)
    for i in range(n_conv):
        for j in range(max(0, i - n_propagator + 1), min(i + 1, n_artery)):
            conv_h2o[i] += propagator[i - j] * artery_h2o[j]
            conv_o2[i] += propagator[i - j] * artery_o2[j]
    rho2 = m * f * (conv_h2o + oef * conv_o2)

    # compartment 1
    # v_post = 0.83*v1
    # v_cap = 0.01*v1
    # R = 0.85  # ratio of small-vessel to large-vessel Hct needed when v1 := CBV * R

    rho1 = v1 * (1 - oef * v_post_cap) * artery_o2

    # package compartments

    rho_ideal = rho1[:n_times] + rho2[:n_times]  # rho_ideal is interpolated to the input function times
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


class Mintun1984Solver(TissueSolver):
    """Solver implementing the Mintun 1984 tissue model for PET data analysis.

    This class implements the tissue model described in Mintun et al. 1984 [1]_ for analyzing
    PET data using dynamic nested sampling. The model accounts for oxygen extraction fraction (OEF),
    water fraction (f_H2O), vascular volume (v_p + 0.5v_c), time offset (t0), arterial dispersion (τa),
    and venous dispersion (τd).

    Args:
        context: Context object containing PET data and configuration.

    Attributes:
        context: Reference to the context object.
        data: Reference to the context's data object containing PET measurements.

    Example:
        >>> context = TissueContext(data_dict)
        >>> solver = Mintun1984Solver(context)
        >>> results = solver.run_nested()
        >>> qm, ql, qh = solver.quantile(results)

    References:
        .. [1] Mintun MA, Raichle ME, Martin WR, Herscovitch P.
               Brain oxygen utilization measured with O-15 radiotracers and positron emission tomography.
               J Nucl Med. 1984 Feb;25(2):177-87. PMID: 6610032.
    """
    def __init__(self, context):
        super().__init__(context)

    @property
    def labels(self):
        return [
            r"OEF", r"$f_{H_2O}$", r"$v_p + \frac{1}{2} v_c$", r"$t_0$", r"$\tau_d$", r"$\sigma$"
        ]

    @staticmethod
    def _loglike(selected_data: dict):
        # Cache values using nonlocal for faster access
        rho = selected_data["rho"]
        timesMid = selected_data["timesMid"]
        taus = selected_data["taus"]
        rho_input_func_interp = selected_data["rho_input_func_interp"]
        ks = selected_data["ks"]
        v1 = selected_data["v1"]
        isidif = selected_data["isidif"]

        # Check dimensions
        if rho.ndim != 1:
            raise ValueError("rho must be 1-dimensional")
        if rho_input_func_interp.ndim != 1:
            raise ValueError("rho_input_func_interp must be 1-dimensional")
        if ks.ndim != 1:
            raise ValueError("ks must be 1-dimensional")
        if not np.isscalar(v1):
            raise ValueError("v1 must be scalar")

        # Create wrapper that matches dynesty's expected signature
        def wrapped_loglike(v):
            nonlocal rho, timesMid, taus, rho_input_func_interp, ks, v1, isidif
            return loglike(
                v,
                rho,
                timesMid,
                taus,
                rho_input_func_interp,
                ks,
                v1,
                isidif)
        return wrapped_loglike

    @staticmethod
    def _prior_transform(selected_data: dict):
        # Create wrapper that matches dynesty's expected signature
        sigma = selected_data["sigma"]

        def wrapped_prior_transform(u):
            nonlocal sigma
            return prior_transform(u, sigma)
        return wrapped_prior_transform

    @staticmethod
    def _run_nested(selected_data: dict) -> dyutils.Results:
        if selected_data["resume"]:
            sampler = dynesty.DynamicNestedSampler.restore(selected_data["checkpoint_file"])
        else:
            loglike = Mintun1984Solver._loglike(selected_data)
            prior_transform = Mintun1984Solver._prior_transform(selected_data)
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
                "ks": self.data.ks[pidx],
                "v1": self.data.v1[pidx],
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
        # _results = [Mintun1984Solver.__run_nested(*arg) for arg in args]
        # self._set_cached_dynesty_results(_results)
        # return _results

        # Use multiprocessing Pool to parallelize execution is incompatible with instance methods
        with Pool() as p:
            _results = p.starmap(Mintun1984Solver._run_nested, [(arg,) for arg in args])
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
        selected_data = {
            "rho": self.data.rho[parc_index],
            "timesMid": self.data.timesMid,
            "taus": self.data.taus,
            "rho_input_func_interp": self.data.rho_input_func_interp,
            "ks": self.data.ks[parc_index],
            "v1": self.data.v1[parc_index],
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

        _results = Mintun1984Solver._run_nested(selected_data)
        self._set_cached_dynesty_results(_results)
        return _results

    def signalmodel(
            self,
            v: list | tuple | NDArray,
            parc_index: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ parc_index selects ks and v1"""

        v = np.array(v, dtype=float)
        if not isinstance(v, np.ndarray) or v.ndim != 1 or len(v) != self.ndim:
            raise ValueError(f"v must be 1-dimensional array of length {self.ndim}")
        if parc_index is None:
            raise ValueError("parc_index must be provided")
        return signalmodel(
            v,
            self.data.timesMid,
            self.data.taus,
            self.data.rho_input_func_interp,
            self.data.ks[parc_index],
            self.data.v1[parc_index],
            self.data.isidif
        )

    def loglike(
            self,
            v: list | tuple | NDArray,
            parc_index: int | None = None
    ) -> float:
        """ parc_index selects ks and v1"""

        v = np.array(v, dtype=float)
        if not isinstance(v, np.ndarray) or v.ndim != 1 or len(v) != self.ndim:
            raise ValueError(f"v must be 1-dimensional array of length {self.ndim}")
        if parc_index is None:
            raise ValueError("parc_index must be provided")
        return loglike(
            v,
            self.data.rho,
            self.data.timesMid,
            self.data.taus,
            self.data.rho_input_func_interp,
            self.data.ks[parc_index],
            self.data.v1[parc_index],
            self.data.isidif
        )
