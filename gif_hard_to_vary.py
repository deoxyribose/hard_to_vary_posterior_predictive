"""Animated version of plot_hard_to_vary.py: smoothly sweep one knob in
each model and render a side-by-side gif. Loads cached SVI samples.
"""

import pickle

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from simulate import simulate
from physics_model import INIT_KNOWN
from fourier_model import design_matrix
from constants import PLANETS, SEMI_MAJOR_AXIS

CACHE_PATH = "/home/frans/htv/svi_samples.pkl"
COLORS = ["tab:gray", "gold", "tab:blue", "tab:red"]
PRIOR_SIGMA_LOG_G = 0.1
TAU = 0.224
N_FRAMES = 60  # full bounce -> visually smooth at ~30 ms / frame


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    samples = cache["samples"]
    G_post = float(np.exp(samples["log_G"].mean()))
    M_post = np.exp(samples["log_M"].mean(axis=0))

    from simulate import make_train_test
    ts_train, obs_train, _, _, _ = make_train_test(seed=0)
    ts_full = jnp.linspace(0.0, 10.0, 600)

    # Bounce δ ∈ [-2, +2, -2] over N_FRAMES (smooth cosine ramp).
    deltas = 2.0 * np.cos(np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False))

    # ---------- Precompute physics trajectories (one per frame) ----------
    print(f"precomputing {N_FRAMES} physics simulations...")
    physics_trajs = []
    for d in deltas:
        G_p = G_post * float(np.exp(d * PRIOR_SIGMA_LOG_G))
        traj = np.asarray(simulate(jnp.float64(G_p), jnp.asarray(M_post),
                                   INIT_KNOWN, ts_full, t0=0.0))
        physics_trajs.append(traj)

    # ---------- Precompute Fourier curves (only Earth-x changes) ----------
    K = 10
    sigma_obs = 0.005
    earth_idx = 2
    period_e = SEMI_MAJOR_AXIS[earth_idx] ** 1.5
    Phi_train_e = np.asarray(design_matrix(ts_train, period_e, K))
    Phi_full_e = np.asarray(design_matrix(ts_full, period_e, K))
    y_train_e_x = np.asarray(obs_train[:, earth_idx, 0])
    D = 2 * K + 1
    A_e = (1.0 / sigma_obs ** 2) * Phi_train_e.T @ Phi_train_e + (1.0 / TAU ** 2) * np.eye(D)
    beta_post_e_x = np.linalg.solve(A_e, (1.0 / sigma_obs ** 2) * Phi_train_e.T @ y_train_e_x)
    knob_idx = 1  # cos(2π t / T_e)
    fourier_post_curves = np.zeros((len(ts_full), 4, 2))
    for i in range(4):
        period = SEMI_MAJOR_AXIS[i] ** 1.5
        Phi_tr = np.asarray(design_matrix(ts_train, period, K))
        Phi_full = np.asarray(design_matrix(ts_full, period, K))
        for c in range(2):
            y_tr = np.asarray(obs_train[:, i, c])
            A = (1.0 / sigma_obs ** 2) * Phi_tr.T @ Phi_tr + (1.0 / TAU ** 2) * np.eye(D)
            beta = np.linalg.solve(A, (1.0 / sigma_obs ** 2) * Phi_tr.T @ y_tr)
            fourier_post_curves[:, i, c] = Phi_full @ beta

    fourier_x_per_frame = []
    for d in deltas:
        beta_p = beta_post_e_x.copy()
        beta_p[knob_idx] += d * TAU
        fourier_x_per_frame.append(Phi_full_e @ beta_p)

    # ---------- Set up figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    physics_lines = []
    for i in range(4):
        line, = axes[0].plot([], [], color=COLORS[i], lw=1.6, label=PLANETS[i])
        physics_lines.append(line)
    fourier_lines = []
    for i in range(4):
        line, = axes[1].plot(fourier_post_curves[:, i, 0], fourier_post_curves[:, i, 1],
                             color=COLORS[i], lw=1.6 if i != earth_idx else 0.6, alpha=0.6 if i != earth_idx else 0.4,
                             label=PLANETS[i])
        fourier_lines.append(line)
    earth_x_line, = axes[1].plot([], [], color=COLORS[earth_idx], lw=2.0)

    delta_text_p = axes[0].text(0.02, 0.97, "", transform=axes[0].transAxes,
                                fontsize=11, va="top", family="monospace")
    delta_text_f = axes[1].text(0.02, 0.97, "", transform=axes[1].transAxes,
                                fontsize=11, va="top", family="monospace")

    for ax, title in zip(axes, [
        "Physics: sweep log G across ±2 prior σ",
        "Fourier: sweep one coefficient (Earth's cos-1) across ±2 prior σ",
    ]):
        ax.plot(0, 0, marker="*", color="orange", markersize=16)
        ax.set_aspect("equal")
        ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9)
        ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Hard-to-vary asymmetry: one knob, six planets warp coherently  vs.  one knob, one curve moves",
        fontsize=12,
    )
    fig.tight_layout()

    def update(frame):
        d = deltas[frame]
        traj = physics_trajs[frame]
        for i, line in enumerate(physics_lines):
            line.set_data(traj[:, i, 0], traj[:, i, 1])
        # Fourier: only Earth-x moves; combine with unperturbed Earth-y
        earth_x_line.set_data(fourier_x_per_frame[frame], fourier_post_curves[:, earth_idx, 1])
        delta_text_p.set_text(f"log_G shift = {d:+.2f} prior σ")
        delta_text_f.set_text(f"β shift     = {d:+.2f} prior σ")
        return physics_lines + [earth_x_line, delta_text_p, delta_text_f]

    print(f"rendering gif ({N_FRAMES} frames)...")
    anim = FuncAnimation(fig, update, frames=N_FRAMES, blit=False, interval=50)
    out = "/home/frans/htv/hard_to_vary.gif"
    anim.save(out, writer=PillowWriter(fps=20))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
