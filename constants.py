"""Solar-system constants in AU / year / M_sun units."""

import numpy as np

PLANETS = ("Mercury", "Venus", "Earth", "Mars")

SEMI_MAJOR_AXIS = np.array([0.387, 0.723, 1.000, 1.524])  # AU

MASSES = np.array([1.66e-7, 2.45e-6, 3.00e-6, 3.23e-7])  # M_sun

G_AU_YR = 4.0 * np.pi ** 2  # AU^3 / yr^2 / M_sun
M_SUN = 1.0


def initial_state_circular():
    """Return the 16-dim coplanar circular initial state.

    Layout per planet i: [x_i, y_i, vx_i, vy_i]; all four planets concatenated.
    """
    a = SEMI_MAJOR_AXIS
    v = np.sqrt(G_AU_YR * M_SUN / a)
    state = np.zeros(16)
    for i in range(4):
        state[4 * i + 0] = a[i]
        state[4 * i + 1] = 0.0
        state[4 * i + 2] = 0.0
        state[4 * i + 3] = v[i]
    return state
