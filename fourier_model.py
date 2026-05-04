"""Bayesian linear regression on a per-planet sin/cos basis.

For each planet i and coordinate c in {x, y}:
    y_n  = phi(t_n)' beta + eps_n,    eps ~ N(0, sigma^2)
    beta ~ N(0, tau^2 I)
    phi(t) = [1, cos(2 pi t/T_i), sin(...), cos(2 pi K t/T_i), sin(...)]

The single-series marginal likelihood and posterior predictive at held-out
times are both closed form (standard Bishop PRML §3.5).
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from constants import SEMI_MAJOR_AXIS


def design_matrix(ts, period, K):
    """Return (N, 2K+1) Fourier design matrix for a given period."""
    omega = 2.0 * jnp.pi / period
    ks = jnp.arange(1, K + 1)
    angles = ts[:, None] * omega * ks[None, :]  # (N, K)
    cols = [jnp.ones_like(ts)[:, None], jnp.cos(angles), jnp.sin(angles)]
    return jnp.concatenate(cols, axis=1)  # (N, 2K+1)


def log_evidence_blr(Phi, y, sigma, tau):
    """Closed-form log-evidence for y ~ N(Phi beta, sigma^2 I), beta ~ N(0, tau^2 I)."""
    N, D = Phi.shape
    # M = sigma^2/tau^2 I + Phi'Phi    (D x D)
    M = (sigma ** 2 / tau ** 2) * jnp.eye(D) + Phi.T @ Phi
    L = jnp.linalg.cholesky(M)
    Phi_y = Phi.T @ y                              # (D,)
    z = jax.scipy.linalg.solve_triangular(L, Phi_y, lower=True)
    yty = jnp.dot(y, y)
    quad = (yty - jnp.dot(z, z)) / sigma ** 2

    # log |C| where C = sigma^2 I + tau^2 Phi Phi'
    # |C| = sigma^{2N} * |I + (tau^2/sigma^2) Phi'Phi| (matrix det. lemma)
    #     = sigma^{2(N-D)} * tau^{2D} * |M|
    # so log|C| = 2(N-D) log sigma + 2 D log tau + 2 sum log diag(L)
    log_det_C = (
        2.0 * (N - D) * jnp.log(sigma)
        + 2.0 * D * jnp.log(tau)
        + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    )

    return -0.5 * (quad + log_det_C + N * jnp.log(2.0 * jnp.pi))


def posterior_mean_predictions(Phi, y, sigma, tau):
    """MAP / posterior-mean fit for visualisation."""
    D = Phi.shape[1]
    M = (sigma ** 2 / tau ** 2) * jnp.eye(D) + Phi.T @ Phi
    beta_hat = jnp.linalg.solve(M, Phi.T @ y)
    return Phi @ beta_hat, beta_hat


def posterior_predictive_logpdf(Phi_train, y_train, Phi_test, y_test, sigma, tau):
    """log p(y_test | y_train) for one Bayesian linear regression series.

    Posterior over beta given y_train is Gaussian:
        Sigma_post = (sigma^-2 Phi'Phi + tau^-2 I)^-1
        mu_post    = sigma^-2 Sigma_post Phi'y_train
    Predictive at test points:
        y_test ~ N(Phi_test mu_post, sigma^2 I + Phi_test Sigma_post Phi_test')
    """
    D = Phi_train.shape[1]
    A = (1.0 / sigma ** 2) * Phi_train.T @ Phi_train + (1.0 / tau ** 2) * jnp.eye(D)
    L = jnp.linalg.cholesky(A)
    rhs = (1.0 / sigma ** 2) * (Phi_train.T @ y_train)
    mu_post = jax.scipy.linalg.cho_solve((L, True), rhs)

    pred_mean = Phi_test @ mu_post                                # (T_test,)
    # Predictive cov = sigma^2 I + Phi_test A^-1 Phi_test'
    Z = jax.scipy.linalg.cho_solve((L, True), Phi_test.T)         # (D, T_test)
    pred_cov = sigma ** 2 * jnp.eye(Phi_test.shape[0]) + Phi_test @ Z

    diff = y_test - pred_mean
    L_pred = jnp.linalg.cholesky(pred_cov)
    z = jax.scipy.linalg.solve_triangular(L_pred, diff, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L_pred)))
    n = y_test.shape[0]
    return -0.5 * (jnp.dot(z, z) + logdet + n * jnp.log(2.0 * jnp.pi))


def fourier_predictive_logpy_test(ts_train, obs_train, ts_test, obs_test,
                                  sigma=0.005, K=10, tau=2.0):
    """Sum log p(y_test | y_train) across all 8 (planet, coord) series, and
    return per-planet posterior-mean predictions on ts_test for plotting."""
    total = 0.0
    pred_means = np.zeros(obs_test.shape)
    for i in range(4):
        period = SEMI_MAJOR_AXIS[i] ** 1.5
        Phi_tr = design_matrix(ts_train, period, K)
        Phi_te = design_matrix(ts_test, period, K)
        for c in range(2):
            y_tr = obs_train[:, i, c]
            ll = float(posterior_predictive_logpdf(
                Phi_tr, y_tr, Phi_te, obs_test[:, i, c], sigma, tau
            ))
            total += ll
            D = Phi_tr.shape[1]
            A = (1.0 / sigma ** 2) * Phi_tr.T @ Phi_tr + (1.0 / tau ** 2) * jnp.eye(D)
            mu = jnp.linalg.solve(A, (1.0 / sigma ** 2) * Phi_tr.T @ y_tr)
            pred_means[:, i, c] = np.asarray(Phi_te @ mu)
    return total, pred_means


def fit_fourier(ts, obs, sigma=0.005, K=10, tau=2.0):
    """Fit per-planet/per-coordinate Bayesian Fourier regression.

    obs: (N, 4, 2). Returns total log-evidence (sum over the 8 series).
    """
    total_log_Z = 0.0
    fits = jnp.zeros_like(obs)
    fits_arr = np.zeros(obs.shape)
    log_Zs = []
    for i in range(4):
        period = SEMI_MAJOR_AXIS[i] ** 1.5
        Phi = design_matrix(ts, period, K)
        for c in range(2):
            y = obs[:, i, c]
            log_Z = float(log_evidence_blr(Phi, y, sigma, tau))
            yhat, _ = posterior_mean_predictions(Phi, y, sigma, tau)
            fits_arr[:, i, c] = np.asarray(yhat)
            total_log_Z += log_Z
            log_Zs.append(log_Z)
    return total_log_Z, fits_arr, log_Zs


def optimise_tau(ts, obs, sigma=0.005, K=10):
    """Empirical-Bayes optimisation of a single shared tau via grid search."""
    taus = jnp.geomspace(0.05, 20.0, 25)
    best_logZ = -jnp.inf
    best_tau = 1.0
    for tau in taus:
        logZ, _, _ = fit_fourier(ts, obs, sigma=sigma, K=K, tau=float(tau))
        if logZ > best_logZ:
            best_logZ = logZ
            best_tau = float(tau)
    return best_tau, best_logZ


if __name__ == "__main__":
    from simulate import make_dataset

    ts, obs, traj = make_dataset()
    tau_star, logZ_star = optimise_tau(ts, obs, sigma=0.005, K=10)
    print(f"best tau = {tau_star:.3f}, log_Z = {logZ_star:.2f}")
