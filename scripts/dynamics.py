import math

import numpy as np

# Disturbances


class ConstantDisturbance:
    def __init__(self, F0):
        self.F0 = F0

    def force(self, t, x, v):
        return self.F0


class SineDisturbance:
    def __init__(self, A, w, phi=0):
        self.A = A
        self.w = w
        self.phi = phi

    def force(self, t, x, v):
        return self.A * math.sin(self.w * t + self.phi)


class ImpulseDisturbance:
    def __init__(self, t0, duration, A):
        self.t0 = t0
        self.t1 = t0 + duration
        self.A = A

    def force(self, t, x, v):
        if self.t0 <= t <= self.t1:
            return self.A
        return 0.0


class OUProcess:
    def __init__(self, theta, sigma, dt):
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = 0.0

    def force(self, t, x, v):
        # state influences drift (this is the key change)
        mean = 0.1 * v + 0.05 * x  # coupling to system

        dw = (
            -self.theta * (self.state - mean) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn()
        )

        self.state += dw
        return self.state


class CompositeDisturbance:
    def __init__(self, disturbances):
        self.disturbances = disturbances

    def force(self, t, x, v):
        total = 0.0
        for d in self.disturbances:
            total += d.force(t, x, v)
        return total


def mass_spring_damper(
    m=1.0,
    c=0.4,
    k=2.0,
    x0=1.0,
    v0=0.0,
    T=10.0,
    dt=0.01,
    u_func=None,
    seed=None,
):
    """
    Simulates a mass-spring-damper system:
        m*x_ddot + c*x_dot + k*x = u + d(t)

    Returns:
        t, x, x_dot, u
    """

    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)

    t = np.linspace(0, T, n_steps)

    x = np.zeros(n_steps)
    v = np.zeros(n_steps)  # velocity
    u = np.zeros(n_steps)

    x[0] = x0
    v[0] = v0

    for i in range(n_steps - 1):
        # default input if none provided
        ui = u_func(t[i]) if u_func is not None else np.random.uniform(-1.0, 1.0)

        u[i] = ui

        # disturbance
        disturbance_model = CompositeDisturbance(
            [
                ConstantDisturbance(F0=1.5),  # steady-state error
                SineDisturbance(A=2.0, w=1.2),  # oscillatory forcing
                OUProcess(theta=2.0, sigma=0.8, dt=0.01),  # correlated noise
                ImpulseDisturbance(t0=7.0, duration=0.5, A=5.0),  # shock event
            ]
        )
        disturbance = disturbance_model.force(t[i], x[i], v[i])

        # dynamics
        a = (ui - c * v[i] - k * x[i] + disturbance) / m

        # Euler integration
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i] * dt

    u[-1] = u[-2]

    return t, x, v, u


def generate_dataset(n_samples=200):

    X = []
    Y = []

    for _ in range(n_samples):
        m = np.random.uniform(0.5, 2.0)
        c = np.random.uniform(0.1, 1.0)
        k = np.random.uniform(1.0, 5.0)

        t, x, v, u = mass_spring_damper(
            m=m, c=c, k=k, T=100, dt=0.01, disturbance_std=0.0
        )

        # feature vector (simple flattening)
        features = np.concatenate([x, v, u])

        theta = np.array([1 / m, -c / m, -k / m])

        X.append(features)
        Y.append(theta)

    return np.array(X), np.array(Y)
