"""End-to-end pipeline:
    1. Simulate true 4-body solar-system dynamics with diffrax, generate noisy
       observations, hold out 25 % as test.
    2. Fit the physics model (Bayesian, with priors on G, masses, ICs) via SVI
       with a multivariate-Normal guide.
    3. Fit a per-planet Fourier basis-function regression in closed form.
    4. Compute log p(y_test | y_train) for both models and compare.
"""

import os
import pickle
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from simulate import make_train_test, simulate
from physics_model import (
    run_svi, predictive_loglik_physics,
    INIT_KNOWN, LOG_M_KNOWN, LOG_G_KNOWN,
)
from fourier_model import fourier_predictive_logpy_test, optimise_tau
from constants import PLANETS


CACHE_PATH = "/home/frans/htv/svi_samples.pkl"


def run_or_load_svi(ts_train, obs_train, num_steps=2000, num_post_samples=400, force=False):
    if not force and os.path.exists(CACHE_PATH):
        print(f"loading cached SVI samples from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print(f"running SVI: num_steps={num_steps}, num_post_samples={num_post_samples}")
    t0 = time.time()
    svi_result, guide, samples = run_svi(
        ts_train, obs_train, num_steps=num_steps,
        num_post_samples=num_post_samples, progress=False,
    )
    elapsed = time.time() - t0
    samples = {k: np.asarray(v) for k, v in samples.items()}
    losses = np.asarray(svi_result.losses)
    print(f"SVI done in {elapsed:.1f}s; final ELBO loss = {float(losses[-1]):.1f}")

    out = dict(samples=samples, losses=losses, elapsed=elapsed)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(out, f)
    print(f"cached SVI samples to {CACHE_PATH}")
    return out


def main():
    ts_train, obs_train, ts_test, obs_test, traj_true = make_train_test(seed=0)
    print(f"data: train {ts_train.shape}, test {ts_test.shape}")

    # ---------- Physics: SVI on train ----------
    print("\n=== Physics SVI (multivariate-Normal guide) ===")
    svi_out = run_or_load_svi(ts_train, obs_train)
    samples = svi_out["samples"]

    # posterior summary
    print("\nposterior summary (mean +/- std):")
    print(f"  log_G    : {samples['log_G'].mean():+.5f} +/- {samples['log_G'].std():.5f}  (true {LOG_G_KNOWN:+.5f})")
    log_M_true = np.asarray(LOG_M_KNOWN)
    for i, name in enumerate(PLANETS):
        m = samples['log_M'][:, i]
        print(f"  log_M[{name:7s}]: {m.mean():+.4f} +/- {m.std():.4f}  (true {log_M_true[i]:+.4f})")
    print(f"  log_sigma: {samples['log_sigma'].mean():+.4f} +/- {samples['log_sigma'].std():.4f}  (true {np.log(0.005):+.4f})")
    init_mean = samples['init'].mean(axis=0)
    init_true = np.asarray(INIT_KNOWN)
    init_err = np.max(np.abs(init_mean - init_true))
    print(f"  max |init mean - true|: {init_err:.4f}")

    # ---------- Physics: predictive on test ----------
    print("\nevaluating physics posterior predictive on test...")
    t0 = time.time()
    log_lik_per_sample, _ = predictive_loglik_physics(samples, ts_test, obs_test)
    log_lik_per_sample = np.asarray(log_lik_per_sample)
    elapsed = time.time() - t0
    log_py_physics = float(jax.scipy.special.logsumexp(jnp.asarray(log_lik_per_sample))
                           - jnp.log(log_lik_per_sample.shape[0]))
    print(f"  done in {elapsed:.1f}s; log p(y_test | y_train, physics) = {log_py_physics:+.2f}")

    # ---------- Fourier: closed-form on train, predictive on test ----------
    print("\n=== Fourier closed-form (train), predictive on test ===")
    tau_star, _ = optimise_tau(ts_train, obs_train, sigma=0.005, K=10)
    print(f"  empirical-Bayes tau* = {tau_star:.3f}")
    log_py_fourier, fourier_test_means = fourier_predictive_logpy_test(
        ts_train, obs_train, ts_test, obs_test, sigma=0.005, K=10, tau=tau_star
    )
    print(f"  log p(y_test | y_train, fourier) = {log_py_fourier:+.2f}")

    # ---------- Compare ----------
    log_pred_BF = log_py_physics - log_py_fourier
    print("\n=== Held-out predictive log-likelihood comparison ===")
    print(f"  log p(y_test | y_train, physics) = {log_py_physics:+.2f}")
    print(f"  log p(y_test | y_train, fourier) = {log_py_fourier:+.2f}")
    print(f"  log predictive ratio (physics over fourier) = {log_pred_BF:+.2f}")
    if log_pred_BF > 5:
        verdict = "physics decisively preferred"
    elif log_pred_BF > 0:
        verdict = "physics preferred"
    elif log_pred_BF > -5:
        verdict = "fourier preferred"
    else:
        verdict = "fourier decisively preferred"
    print(f"  verdict: {verdict}")

    # ---------- Plots ----------
    G_post = float(np.exp(samples['log_G'].mean()))
    M_post = np.exp(samples['log_M'].mean(axis=0))
    init_post = samples['init'].mean(axis=0)
    ts_full = jnp.concatenate([ts_train, ts_test])
    sort_idx = jnp.argsort(ts_full)
    ts_sorted = ts_full[sort_idx]
    traj_post = np.asarray(simulate(jnp.float64(G_post), jnp.asarray(M_post),
                                    jnp.asarray(init_post), ts_sorted))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    colors = ["tab:gray", "gold", "tab:blue", "tab:red"]
    for i in range(4):
        # physics panel
        ax = axes[0]
        ax.plot(traj_post[:, i, 0], traj_post[:, i, 1], color=colors[i], lw=1.2, label=PLANETS[i])
        ax.scatter(obs_train[:, i, 0], obs_train[:, i, 1], s=5, color=colors[i], alpha=0.25)
        ax.scatter(obs_test[:, i, 0], obs_test[:, i, 1], s=15, color=colors[i],
                   marker="x", linewidth=1.0, label=f"{PLANETS[i]} (test)" if i == 0 else None)
        # fourier panel
        ax = axes[1]
        ax.plot(fourier_test_means[:, i, 0], fourier_test_means[:, i, 1],
                color=colors[i], lw=0.0, marker="o", markersize=4, label=PLANETS[i])
        ax.scatter(obs_train[:, i, 0], obs_train[:, i, 1], s=5, color=colors[i], alpha=0.25)
        ax.scatter(obs_test[:, i, 0], obs_test[:, i, 1], s=15, color=colors[i],
                   marker="x", linewidth=1.0)

    for ax, title in zip(axes, ["Physics: posterior trajectory + test (x)",
                                 "Fourier: predictive mean on test"]):
        ax.plot(0, 0, marker="*", color="orange", markersize=18)
        ax.set_aspect("equal")
        ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"log p(y_test|y_train): physics = {log_py_physics:+.1f}    "
        f"fourier = {log_py_fourier:+.1f}    "
        f"diff = {log_pred_BF:+.1f}"
    )
    fig.tight_layout()
    fig.savefig("/home/frans/htv/fits.png", dpi=120)
    print("\nsaved /home/frans/htv/fits.png")


if __name__ == "__main__":
    main()
