"""Bayesian physics model: log-normal priors on G and masses, fixed initial
conditions, and a Normal observation likelihood whose mean is the output of
the diffrax N-body simulator. Six latent parameters: log_G, log_M[0..3],
log_sigma.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize as sp_minimize

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.util import log_density
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
    log_sigma = numpyro.sample("log_sigma", dist.Normal(LOG_SIGMA_KNOWN, 1.0))

    G = jnp.exp(log_G)
    masses = jnp.exp(log_M)
    sigma = jnp.exp(log_sigma)
    # INIT_KNOWN is the state at t=0; force the integrator to start there
    # so a sliced ts (test/train) is interpreted correctly.
    traj = simulate(G, masses, INIT_KNOWN, ts, t0=0.0)
    pred = traj[..., :2]
    # Diffrax (throw=False) returns NaN when the integrator hits its step limit.
    # Replace NaN here so the Normal validates; optax.zero_nans in the SVI
    # optimizer drops the resulting NaN gradient for that step.
    pred_safe = jnp.nan_to_num(pred, nan=1e6, posinf=1e6, neginf=-1e6)
    numpyro.sample("y", dist.Normal(pred_safe, sigma), obs=obs)


def _known_init_values():
    return {
        "log_G": jnp.float64(LOG_G_KNOWN),
        "log_M": LOG_M_KNOWN,
        "log_sigma": jnp.float64(LOG_SIGMA_KNOWN),
    }


def run_map(ts, obs):
    """MAP via L-BFGS on the flattened parameter vector. Quasi-Newton handles
    the 5-OOM ill-conditioning between log_G and the masses that defeated
    Adam. Returns (scipy_result, map_values_dict).
    """
    init = _known_init_values()
    flat_init, unflatten = ravel_pytree(init)

    @jax.jit
    def value_and_grad(flat):
        def loss(f):
            lp, _ = log_density(numpyro_model, (ts, obs), {}, unflatten(f))
            return -lp
        return jax.value_and_grad(loss)(flat)

    def f_np(x):
        v, g = value_and_grad(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=np.float64)

    res = sp_minimize(
        f_np, np.asarray(flat_init), method="L-BFGS-B", jac=True,
        options=dict(gtol=1e-6, maxiter=200),
    )
    map_values = unflatten(jnp.asarray(res.x))
    map_values = {k: jnp.asarray(v) for k, v in map_values.items()}
    return res, map_values


def run_svi(ts, obs, init_values=None, num_steps=1500, lr=1e-4, seed=1,
            num_post_samples=400, num_particles=8, progress=True):
    """SVI with a multivariate-normal guide.

    `init_values` defaults to the known parameters; pass MAP estimates here
    to anchor the guide near the mode. `num_particles>1` averages the ELBO
    estimator over multiple MC samples per step, which is essential here:
    a single bad guide draw makes the ODE integrate into garbage and
    log p(y|theta) blows up, destabilising the optimizer.
    """
    if init_values is None:
        init_values = _known_init_values()
    guide = AutoMultivariateNormal(
        numpyro_model, init_loc_fn=init_to_value(values=init_values), init_scale=1e-3
    )
    optimizer = numpyro.optim.optax_to_numpyro(
        optax.chain(optax.zero_nans(), optax.clip_by_global_norm(1.0), optax.adam(lr))
    )
    svi = SVI(numpyro_model, guide, optimizer,
              loss=Trace_ELBO(num_particles=num_particles, vectorize_particles=True))
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
    log_sigma = jnp.asarray(samples["log_sigma"])

    def one(log_g, log_m, log_s):
        G = jnp.exp(log_g)
        m = jnp.exp(log_m)
        sigma = jnp.exp(log_s)
        traj = simulate(G, m, INIT_KNOWN, ts_test, t0=0.0)
        pred = traj[..., :2]
        log_pt = -0.5 * jnp.log(2 * jnp.pi) - log_s - 0.5 * ((obs_test - pred) / sigma) ** 2
        return log_pt.sum(), log_pt

    return jax.vmap(one)(log_G, log_M, log_sigma)


def predictive_logpy_test(log_lik_per_sample):
    """log p(y_test | y_train) = logsumexp(log p(y_test|theta_s)) - log S."""
    S = log_lik_per_sample.shape[0]
    return float(jax.scipy.special.logsumexp(log_lik_per_sample) - jnp.log(S))
