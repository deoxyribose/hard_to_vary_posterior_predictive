"""N-body integration of inner planets with the Sun pinned at the origin."""

from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

from constants import G_AU_YR, M_SUN, MASSES, initial_state_circular


def _accelerations(positions, G, masses):
    """positions: (4, 2). Returns (4, 2) accelerations."""
    r = jnp.linalg.norm(positions, axis=-1, keepdims=True)
    a_sun = -G * M_SUN * positions / r ** 3

    diff = positions[None, :, :] - positions[:, None, :]  # (i, j, 2): r_j - r_i
    dist2 = jnp.sum(diff * diff, axis=-1, keepdims=True)
    # Mask diagonal with a safe value so gradients don't explode at d=0,
    # then zero out the diagonal contribution after the divide.
    safe_dist2 = dist2 + jnp.eye(positions.shape[0])[..., None]  # add 1 on diagonal
    inv_cube = safe_dist2 ** (-1.5)
    off_diag = 1.0 - jnp.eye(positions.shape[0])[..., None]
    pairwise = G * masses[None, :, None] * diff * inv_cube * off_diag  # (i, j, 2)
    a_planets = pairwise.sum(axis=1)
    return a_sun + a_planets


def nbody_rhs(t, y, args):
    G, masses = args
    state = y.reshape(4, 4)
    pos = state[:, :2]
    vel = state[:, 2:]
    acc = _accelerations(pos, G, masses)
    out = jnp.concatenate([vel, acc], axis=1)
    return out.reshape(16)


@partial(jax.jit, static_argnames=("max_steps",))
def simulate(G, masses, init_state, ts, t0=None, max_steps=20_000):
    """Integrate the N-body ODE from `t0` (default ts[0]) up to ts[-1] and
    save at the times in `ts`. `init_state` is the state at `t0`, so when
    inferring on a slice of times that does not start at 0, callers should
    pass `t0=0.0` so the known initial condition is interpreted correctly.
    """
    if t0 is None:
        t0 = ts[0]
    term = diffrax.ODETerm(nbody_rhs)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=1e-7, atol=1e-9)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=ts[-1],
        dt0=0.005,
        y0=init_state,
        args=(G, masses),
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=controller,
        max_steps=max_steps,
        throw=False,  # don't raise; return NaN on failure
    )
    return sol.ys.reshape(-1, 4, 4)


def make_dataset(seed=0, T_years=10.0, n_obs=200, sigma=0.005):
    ts = jnp.linspace(0.0, T_years, n_obs)
    init_state = jnp.asarray(initial_state_circular())
    masses = jnp.asarray(MASSES)
    traj = simulate(G_AU_YR, masses, init_state, ts)
    positions = traj[..., :2]  # (n_obs, 4, 2)
    key = jax.random.key(seed)
    noise = sigma * jax.random.normal(key, positions.shape)
    obs = positions + noise
    return ts, obs, traj


def make_train_test(seed=0, T_years=10.0, n_obs=200, sigma=0.005, test_frac=0.25):
    """Generate ts/obs and split into train/test (every k-th sample held out)."""
    ts, obs, traj = make_dataset(seed=seed, T_years=T_years, n_obs=n_obs, sigma=sigma)
    # interleaved split: every 4th observation goes to test (when test_frac=0.25)
    step = int(round(1 / test_frac))
    n = ts.shape[0]
    idx = jnp.arange(n)
    test_mask = (idx % step) == 0
    train_mask = ~test_mask
    return (
        ts[train_mask], obs[train_mask],
        ts[test_mask], obs[test_mask],
        traj,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ts, obs, traj = make_dataset()
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["tab:gray", "gold", "tab:blue", "tab:red"]
    names = ["Mercury", "Venus", "Earth", "Mars"]
    ax.plot(0, 0, marker="*", color="orange", markersize=18, label="Sun")
    for i in range(4):
        ax.plot(traj[:, i, 0], traj[:, i, 1], color=colors[i], lw=1.0, label=names[i])
        ax.scatter(obs[:, i, 0], obs[:, i, 1], s=4, color=colors[i], alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_title("Synthetic data: inner planets, 10 yr, sigma=0.005 AU")
    ax.legend()
    fig.tight_layout()
    fig.savefig("orbits.png", dpi=120)
    print("saved orbits.png; trajectory shape:", traj.shape)
