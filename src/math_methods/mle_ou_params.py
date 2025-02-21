import typing as tp
import numpy as np


def estimate_params_MLE(X: np.ndarray, dt: float = 0.01) -> tp.Tuple[float, float, float]:
    """
    A function that calculates maximum likelihood estimates for parameters of a stochastic process
    
    :param X: modeling data from stochastic process
    :param dt: Time step size for Euler-Maruyama discretization
    :return: Estimation of stochastic process parameters
    """
    N = X.size
    Xx  = np.sum(X[0:-1])
    Xy  = np.sum(X[1:])
    Xxx = np.sum(X[0:-1]**2)
    Xxy = np.sum(X[0:-1] * X[1:])
    Xyy = np.sum(X[1:]**2)

    mu_hat = (Xy * Xxx - Xx * Xxy) /  \
        (N * (Xxx - Xxy) - (Xx**2 - Xx * Xy) )

    theta_hat = (Xxy - mu_hat * Xx - mu_hat * Xy + N * mu_hat**2) / \
        (Xxx - 2 * mu_hat * Xx + N * mu_hat**2)
    theta_hat = -1 / dt * np.log(theta_hat)

    prefactor = 2 * theta_hat / (N*(1-np.exp(-2*theta_hat*dt)))
    term = Xyy - 2*np.exp(-theta_hat*dt) * Xxy + np.exp(-2*theta_hat*dt) * Xxx - 2*mu_hat*(1-np.exp(-theta_hat*dt)) * (Xy - Xx * np.exp(-theta_hat*dt)) + N * mu_hat**2* ( 1-np.exp(-theta_hat * dt))**2

    sigma_hat = np.sqrt(prefactor * term)
    return theta_hat, mu_hat, sigma_hat