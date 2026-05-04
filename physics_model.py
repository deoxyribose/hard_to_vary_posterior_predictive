"""Bayesian physics model: log-normal priors on G and masses, broad normals on
initial conditions, and a Normal observation likelihood whose mean is the
output of the diffrax N-body simulator.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal
import optax

from constants import G_AU_YR, MASSES, initial_state_circular
from simulate import simulate


LOG_G_KNOWN = float(np.log(G_AU_YR))
LOG_M_KNOWN = jnp.asarray(np.log(MASSES))
INIT_KNOWN = jnp.asarray(initial_state_circular())
LOG_SIGMA_KNOWN = float(np.log(0.005))


def numpyro_model(ts, obs):
    log_G = numpyro.sample("log_G", dist.Normal(LOG_G_KNOWN, 0.1))
    log_M = numpyro.sample("log_M", dist.Normal(LOG_M_KNOWN, 0.1 * jnp.ones(4)))
    init = numpyro.sample("init", dist.Normal(INIT_KNOWN, 0.05 * jnp.ones(16)))
    log_sigma = numpyro.sample("log_sigma", dist.Normal(LOG_SIGMA_KNOWN, 1.0))

    G = jnp.exp(log_G)
    masses = jnp.exp(log_M)
    sigma = jnp.exp(log_sigma)
    traj = simulate(G, masses, init, ts)
    pred = traj[..., :2]
    # Diffrax (throw=False) returns NaN when the integrator hits its step limit.
    # Replace NaN here so the Normal validates; optax.zero_nans in the SVI
    # optimizer drops the resulting NaN gradient for that step.
    pred_safe = jnp.nan_to_num(pred, nan=1e6, posinf=1e6, neginf=-1e6)
    numpyro.sample("y", dist.Normal(pred_safe, sigma), obs=obs)


def run_svi(ts, obs, num_steps=2000, lr=0.005, seed=0, num_post_samples=400,
            progress=True):
    """SVI with a multivariate-normal guide initialised at the known params."""
    init_values = {
        "log_G": jnp.float64(LOG_G_KNOWN),
        "log_M": LOG_M_KNOWN,
        "init": INIT_KNOWN,
        "log_sigma": jnp.float64(LOG_SIGMA_KNOWN),
    }
    guide = AutoMultivariateNormal(numpyro_model, init_loc_fn=init_to_value(values=init_values))
    optimizer = numpyro.optim.optax_to_numpyro(
        optax.chain(optax.zero_nans(), optax.clip(1e3), optax.adam(lr))
    )
    svi = SVI(numpyro_model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(jax.random.key(seed), num_steps, ts=ts, obs=obs,
                         progress_bar=progress)

    rng = jax.random.key(seed + 1)
    posterior_samples = guide.sample_posterior(
        rng, svi_result.params, sample_shape=(num_post_samples,)
    )
    return svi_result, guide, posterior_samples


def predictive_loglik_physics(samples, ts_test, obs_test):
    """For each posterior sample s, simulate on ts_test and score obs_test.

    Returns log_lik_per_sample (S,) and per-point log-lik (S, T_test, 4, 2).
    """
    log_G = jnp.asarray(samples["log_G"])
    log_M = jnp.asarray(samples["log_M"])
    init = jnp.asarray(samples["init"])
    log_sigma = jnp.asarray(samples["log_sigma"])

    def one(log_g, log_m, init_s, log_s):
        G = jnp.exp(log_g)
        m = jnp.exp(log_m)
        sigma = jnp.exp(log_s)
        traj = simulate(G, m, init_s, ts_test)
        pred = traj[..., :2]
        log_pt = -0.5 * jnp.log(2 * jnp.pi) - log_s - 0.5 * ((obs_test - pred) / sigma) ** 2
        return log_pt.sum(), log_pt

    return jax.vmap(one)(log_G, log_M, init, log_sigma)


def predictive_logpy_test(log_lik_per_sample):
    """log p(y_test | y_train) = logsumexp(log p(y_test|theta_s)) - log S."""
    S = log_lik_per_sample.shape[0]
    return float(jax.scipy.special.logsumexp(log_lik_per_sample) - jnp.log(S))
