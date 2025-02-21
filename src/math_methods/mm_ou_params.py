import typing as tp
import numpy as np


def estimate_params_MM(X: np.ndarray, dt: float = 0.01) -> tp.Tuple[float, float, float]:
    """
    Method of moments for calculating estimates for parameters of a stochastic process
    
    :param X: modeling data from stochastic process
    :param dt: Time step size for Euler-Maruyama discretization
    :return: Estimation of stochastic process parameters
    """
    mu_hat = np.mean(X)
    cov_hat = np.cov(X[:-1], X[1:])[0, 1]
    theta_hat = np.log(np.var(X) / cov_hat) / dt
    sigma_hat = np.sqrt(2 * theta_hat * np.var(X))
    return theta_hat, mu_hat, sigma_hat