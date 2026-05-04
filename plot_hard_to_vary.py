"""Visualise the 'hard to vary' asymmetry between the physics and Fourier
models. Twiddle one knob in each model and show what changes:

  - Physics: shifting log_G alone warps every planet's orbit at once.
  - Fourier: shifting a single coefficient changes only one (planet, coord).

Loads the cached SVI samples produced by run.py.
"""

import pickle

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from simulate import simulate, make_train_test
from physics_model import INIT_KNOWN
from fourier_model import design_matrix
from constants import PLANETS, SEMI_MAJOR_AXIS

CACHE_PATH = "/home/frans/htv/svi_samples.pkl"
COLORS = ["tab:gray", "gold", "tab:blue", "tab:red"]
PRIOR_SIGMA_LOG_G = 0.1


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    samples = cache["samples"]
    G_post = float(np.exp(samples["log_G"].mean()))
    M_post = np.exp(samples["log_M"].mean(axis=0))

    ts_train, obs_train, ts_test, obs_test, _ = make_train_test(seed=0)
    ts_full = jnp.linspace(0.0, 10.0, 600)

    # ---------- Physics: one knob (log_G), four perturbations ----------
    deltas = np.array([-2.0, -1.0, +1.0, +2.0])  # in prior-sigma units
    physics_perturbed = []
    for d in deltas:
        G_p = G_post * float(np.exp(d * PRIOR_SIGMA_LOG_G))
        traj = np.asarray(simulate(jnp.float64(G_p), jnp.asarray(M_post),
                                   INIT_KNOWN, ts_full, t0=0.0))
        physics_perturbed.append(traj)
    traj_post = np.asarray(simulate(jnp.float64(G_post), jnp.asarray(M_post),
                                    INIT_KNOWN, ts_full, t0=0.0))

    # ---------- Fourier: one knob, only one planet/coord changes ----------
    # Refit Earth-x (planet=2, coord=0) to get its posterior mean, then perturb
    # one Fourier coefficient by ±d * tau (Fourier's prior sigma) -- the same
    # 'one prior sigma' yardstick we used for physics's log_G.
    K = 10
    sigma_obs = 0.005
    tau = 0.224  # empirical-Bayes value reported by run.py
    earth_idx = 2
    period_e = SEMI_MAJOR_AXIS[earth_idx] ** 1.5
    Phi_train_e = np.asarray(design_matrix(ts_train, period_e, K))
    Phi_full_e = np.asarray(design_matrix(ts_full, period_e, K))
    y_train_e_x = np.asarray(obs_train[:, earth_idx, 0])
    D = 2 * K + 1
    A_e = (1.0 / sigma_obs ** 2) * Phi_train_e.T @ Phi_train_e + (1.0 / tau ** 2) * np.eye(D)
    beta_post_e_x = np.linalg.solve(A_e, (1.0 / sigma_obs ** 2) * Phi_train_e.T @ y_train_e_x)
    knob_idx = 1  # column 1 = first cosine (period of Earth's orbit)
    fourier_perturbed_x_e = []
    for d in deltas:
        beta_p = beta_post_e_x.copy()
        beta_p[knob_idx] = beta_post_e_x[knob_idx] + d * tau
        fourier_perturbed_x_e.append(Phi_full_e @ beta_p)

    # For the other planets and y-coord, draw their unperturbed posterior mean.
    fourier_post_curves = np.zeros((len(ts_full), 4, 2))
    for i in range(4):
        period = SEMI_MAJOR_AXIS[i] ** 1.5
        Phi_tr = np.asarray(design_matrix(ts_train, period, K))
        Phi_full = np.asarray(design_matrix(ts_full, period, K))
        for c in range(2):
            y_tr = np.asarray(obs_train[:, i, c])
            A = (1.0 / sigma_obs ** 2) * Phi_tr.T @ Phi_tr + (1.0 / tau ** 2) * np.eye(D)
            beta = np.linalg.solve(A, (1.0 / sigma_obs ** 2) * Phi_tr.T @ y_tr)
            fourier_post_curves[:, i, c] = Phi_full @ beta

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    ax = axes[0]
    for i in range(4):
        ax.plot(traj_post[:, i, 0], traj_post[:, i, 1], color=COLORS[i], lw=1.6,
                label=PLANETS[i] if i == 0 or True else None)
        for traj_p in physics_perturbed:
            ax.plot(traj_p[:, i, 0], traj_p[:, i, 1], color=COLORS[i], lw=0.7, alpha=0.35)
    ax.plot(0, 0, marker="*", color="orange", markersize=16)
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
    ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]")
    ax.set_title("Physics: one knob (log G ± 2 prior σ) → ALL 4 orbits warp together")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    for i in range(4):
        ax.plot(fourier_post_curves[:, i, 0], fourier_post_curves[:, i, 1],
                color=COLORS[i], lw=1.6, label=PLANETS[i])
        if i == earth_idx:
            # Only Earth's x-curve gets perturbed; combine with unperturbed y.
            for fp in fourier_perturbed_x_e:
                ax.plot(fp, fourier_post_curves[:, i, 1],
                        color=COLORS[i], lw=0.7, alpha=0.35)
    ax.plot(0, 0, marker="*", color="orange", markersize=16)
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
    ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]")
    ax.set_title("Fourier: one knob (Earth's cos-1st-harmonic) → only Earth-x changes")
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Hard-to-vary asymmetry: physics knobs are coupled, Fourier knobs are independent",
        fontsize=12,
    )
    fig.tight_layout()
    out = "/home/frans/htv/hard_to_vary.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
