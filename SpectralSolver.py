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
    sigma: float,
    M: int
) -> np.ndarray:
    v = u
    v[0] = u[0]
    v[1] = u[1] * 0.05 + 0.00003  # 1/(3*T_{end}) < \beta < 3/T_{in}
    for m in np.arange(1, M):
        v[2 * m] = u[2 * m]  # \alpha_1 ~ K_1
        v_max = v[2 * m - 1] - np.finfo(float).eps
        v[2 * m + 1] = u[2 * m + 1] * v_max  # \beta_1 ~ k_2; \beta_2 < \beta_1
    v[2 * M] = u[2 * M] * 0.05  # \alpha_0 ~ V_p

    T = 2 * M + 1
    S = 2 * M + 2
    v[T] = u[T] * 20  # t_0 (s)
    v[S] = u[S] * sigma  # sigma ~ fraction of A0
    return v


@jit(nopython=True)
def loglike(
    v: np.ndarray,
    rho: np.ndarray,
    timesMid: np.ndarray,
    taus: np.ndarray,
    rho_input_func_interp: np.ndarray,
    M: int,
    isidif: bool
) -> float:
    assert rho.ndim == 1, "rho must be 1-dimensional"
    rho_pred, _, _, _ = signalmodel(v, timesMid, taus, rho_input_func_interp, M, isidif)
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
    M: int,
    isidif: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ SpectralSolver assumes all input params to be decay-corrected """    

    a = np.zeros(M)
    b = np.zeros(M)
    for m in np.arange(M):
        a[m] = v[2 * m]
        b[m] = v[2 * m + 1]
    a_0 = v[2 * M]
    t_0 = v[2 * M + 1]

    n_times = rho_input_func_interp.shape[0]
    timesIdeal = np.arange(0, n_times)
    timesIdeal1 = np.arange(0, n_times + 1)

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
    rho_input_func_interp1 = np.append(rho_input_func_interp, rho_input_func_interp[-1])

    # propagate input function

    propagator = np.zeros((M, n_times))
    for m in np.arange(M):
        propagator[m] = a[m] * np.exp(-b[m] * timesIdeal)

    rho_ideal = 0
    for m in np.arange(M):
        n_propagator = len(propagator[m])
        n_artery = len(rho_input_func_interp) 
        n_conv = n_propagator + n_artery - 1
        conv = np.zeros(n_conv)
        for i in range(n_conv):
            for j in range(max(0, i - n_propagator + 1), min(i + 1, n_artery)):
                conv[i] += propagator[m][i - j] * rho_input_func_interp[j]
        rho_ideal = rho_ideal + conv        
    rho_ideal = rho_ideal[:n_times]
    
    # Implement cumulative trapezoidal integration manually for numba compatibility

    dt = timesIdeal1[1] - timesIdeal1[0]  # Uniform spacing
    cumtrapz = np.zeros(len(timesIdeal))
    cumsum = 0.0
    for i in range(len(cumtrapz)):
        cumsum += 0.5 * (rho_input_func_interp1[i] + rho_input_func_interp1[i+1]) * dt
        cumtrapz[i] = cumsum
    rho_ideal = rho_ideal + a_0 * cumtrapz

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


class SpectralSolver(TissueSolver):
    """Solver implementing spectral methods for PET data analysis.

    This class implements the spectral methods described in Bertoldo et al. 1998 [1] for analyzing
    PET data using dynamic nested sampling. 

    Args:
        context: Context object containing PET data and configuration.

    Attributes:
        context: Reference to the context object.
        data: Reference to the context's data object containing PET measurements.

    Example:
        >>> context = TissueContext(data_dict)
        >>> solver = SpectralSolver(context)
        >>> results = solver.run_nested()
        >>> qm, ql, qh = solver.quantile(results)

    References:

        A. Bertoldo, P. Vicini, G. Sambuceti, A. A. Lammertsma, O. Parodi and C. Cobelli,
        "Evaluation of compartmental and spectral analysis models of [/sup 18/F]FDG kinetics for heart and brain 
        studies with PET"
        IEEE Transactions on Biomedical Engineering, vol. 45, no. 12, pp. 1429-1448, Dec. 1998,
        doi: 10.1109/10.730437.
        keywords: {Spectral analysis;Kinetic theory;Heart;Brain modeling;Positron emission tomography;Sugar;
        Biochemistry;Myocardium;Biomedical informatics;Hospitals}
    """

    def __init__(self, context):
        super().__init__(context)

    @property
    def labels(self):
        if self._labels:
            return self._labels

        M = self.context.M
        lbls = []
        for m in np.arange(M):
            lbls.extend([fr"$\alpha_{{{m + 1}}}$", fr"$\beta_{{{m + 1}}}$"])
        lbls.extend([r"$a_0$", r"$\sigma$"])
        self._labels = lbls
        return lbls
    
    @staticmethod
    def _loglike(selected_data: dict):
        # Cache values using nonlocal for faster access
        rho = selected_data["rho"]
        timesMid = selected_data["timesMid"]
        taus = selected_data["taus"]
        rho_input_func_interp = selected_data["rho_input_func_interp"]
        M = selected_data["M"]
        isidif = selected_data["isidif"]

        # Check dimensions
        if rho.ndim != 1:
            raise ValueError("rho must be 1-dimensional")
        if rho_input_func_interp.ndim != 1:
            raise ValueError("rho_input_func_interp must be 1-dimensional")

        # Create wrapper that matches dynesty's expected signature
        def wrapped_loglike(v):
            nonlocal rho, timesMid, taus, rho_input_func_interp, M, isidif
            return loglike(
                v,
                rho,
                timesMid,
                taus,
                rho_input_func_interp,
                M,
                isidif)
        return wrapped_loglike

    @staticmethod
    def _prior_transform(selected_data: dict):
        # Create wrapper that matches dynesty's expected signature
        sigma = selected_data["sigma"]
        M = selected_data["M"]

        def wrapped_prior_transform(u):
            nonlocal sigma, M
            return prior_transform(u, sigma, M)
        return wrapped_prior_transform

    @staticmethod
    def _run_nested(selected_data: dict) -> dyutils.Results:
        if selected_data["resume"]:
            sampler = dynesty.DynamicNestedSampler.restore(selected_data["checkpoint_file"])
        else:
            loglike = SpectralSolver._loglike(selected_data)
            prior_transform = SpectralSolver._prior_transform(selected_data)
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
                "M": self.data.M,
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
            _results = p.starmap(SpectralSolver._run_nested, [(arg,) for arg in args])
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
            "isidif": self.data.isidif,
            "sigma": self.data.sigma,
            "M": self.data.M,
            "ndim": self.ndim,
            "sample": self.data.sample,
            "nlive": self.data.nlive,
            "rstate": self.data.rstate,
            "checkpoint_file": checkpoint_file,
            "resume": resume,
            "pfrac": self.data.pfrac,
            "print_progress": print_progress
        }

        _results = SpectralSolver._run_nested(selected_data)
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
            self.data.M,
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
            self.data.M,
            self.data.isidif
        )
