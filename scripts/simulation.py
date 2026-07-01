import control
import numpy as np


def simulate_closed_loop(K, m, c, k, T=10, dt=0.01):

    n = int(T / dt)
    t = np.linspace(0, T, n)

    x = np.zeros(n)
    v = np.zeros(n)
    u = np.zeros(n)

    x[0] = 1.0

    for i in range(n - 1):
        state = np.array([x[i], v[i]])

        # control law
        u[i] = -K @ state

        # TRUE dynamics
        a = (u[i] - c * v[i] - k * x[i]) / m

        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i] * dt

    u[-1] = u[-2]

    return t, x, v, u
