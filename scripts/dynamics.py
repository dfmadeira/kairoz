import numpy as np


def mass_spring_damper(
    m=1.0,
    c=0.4,
    k=2.0,
    x0=1.0,
    v0=0.0,
    T=10.0,
    dt=0.01,
    u_func=None,
    disturbance_std=0.0,
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

        # disturbance (white noise)
        disturbance = disturbance_std * np.random.randn()

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
