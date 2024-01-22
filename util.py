# util.py

# basic numeric setup
import numpy as np
from numpy import linalg


def loglikelihood(x):
    # ...do something

    ndim = 3  # number of dimensions
    C = np.identity(ndim)  # set covariance to identity matrix
    C[C == 0] = 0.95  # set off-diagonal terms (strongly correlated)
    Cinv = linalg.inv(C)  # precision matrix
    lnorm = -0.5 * (np.log(2 * np.pi) * ndim + np.log(linalg.det(C)))  # ln(normalization)

    # 3-D correlated multivariate normal log-likelihood
    """Multivariate normal log-likelihood."""
    return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm
