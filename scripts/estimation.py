import numpy as np


class RLS:
    def __init__(self, n_params=3, lam=0.99, delta=1000.0):

        self.n = n_params
        self.lam = lam  # forgetting factor

        self.theta = np.zeros((n_params, 1))

        self.P = np.eye(n_params) * delta

        self.prev_x = None
        self.prev_v = None

        self.dt = 0.01

    def update(self, x, v, u):

        # -------------------------
        # 1. compute acceleration
        # -------------------------
        if self.prev_v is None:
            self.prev_x = x
            self.prev_v = v
            return self.theta.flatten()

        a = (v - self.prev_v) / self.dt

        # regression vector
        phi = np.array([[u], [v], [x]])

        # -------------------------
        # 2. RLS gain
        # -------------------------
        P_phi = self.P @ phi
        gain_den = self.lam + phi.T @ P_phi

        K = P_phi / gain_den

        # -------------------------
        # 3. prediction error
        # -------------------------
        y = np.array([[a]])

        err = y - phi.T @ self.theta

        # -------------------------
        # 4. parameter update
        # -------------------------
        self.theta = self.theta + K @ err

        # -------------------------
        # 5. covariance update
        # -------------------------
        self.P = (self.P - K @ phi.T @ self.P) / self.lam

        # store state
        self.prev_v = v
        self.prev_x = x

        return self.theta.flatten()


class ThetaEstimator:
    def __init__(self, mode="recursive", window_size=200, update_every=1):
        self.mode = mode
        self.window_size = window_size
        self.update_every = update_every

        self.buffer = []
        self.theta = None

    def update(self, i, x, v, u):

        self.buffer.append((x, v, u))

        if i % self.update_every != 0:
            return None

        if self.mode == "recursive":
            return self._recursive_update()

        elif self.mode == "window":
            return self._window_update()

        else:
            raise ValueError("Unknown mode")

    def _window_update(self):
        data = self.buffer[-self.window_size :]

        # reuse your existing estimator here
        # you already have estimate_parameters(...)
        t_dummy = np.arange(len(data)) * 0.01

        x = np.array([d[0] for d in data])
        v = np.array([d[1] for d in data])
        u = np.array([d[2] for d in data])

        theta = estimate_parameters(t_dummy, x, v, u, dt=0.01)

        self.theta = theta
        return theta

    def _recursive_update(self):
        # TEMPORARY: for now just behave like window size = 1
        x, v, u = self.buffer[-1]

        t_dummy = np.array([0.0])
        theta = estimate_parameters(
            t_dummy, np.array([x]), np.array([v]), np.array([u]), dt=0.01
        )

        self.theta = theta
        return theta


def estimate_parameters(t, x, v, u, dt):  # linear
    # approximate acceleration
    a = np.zeros_like(v)
    a[:-1] = (v[1:] - v[:-1]) / dt
    a[-1] = a[-2]

    # regression matrix
    Phi = np.column_stack([u, v, x])  # [u, x_dot, x]
    y = a

    # least squares solution
    theta, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)

    return theta


def estimate_theta_from_trajectory(model, x, v, u):

    features = np.concatenate([x, v, u])
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        theta = model(features).numpy()[0]

    return theta


def compare_estimators(model, n_tests=20):

    ls_errors = []
    nn_errors = []

    for _ in range(n_tests):
        m = np.random.uniform(0.5, 2.0)
        c = np.random.uniform(0.1, 1.0)
        k = np.random.uniform(1.0, 5.0)

        t, x, v, u = simulate_mass_spring_damper(
            m=m, c=c, k=k, u_func=Excitation("prbs")
        )

        theta_true = np.array([1 / m, -c / m, -k / m])

        theta_ls = estimate_parameters(t, x, v, u, dt=0.01)
        theta_nn = estimate_theta_from_trajectory(model, x, v, u)

        ls_error = np.linalg.norm(theta_true - theta_ls)
        nn_error = np.linalg.norm(theta_true - theta_nn)

        ls_errors.append(ls_error)
        nn_errors.append(nn_error)

    print("LS mean error:", np.mean(ls_errors))
    print("NN mean error:", np.mean(nn_errors))

    print("LS std:", np.std(ls_errors))
    print("NN std:", np.std(nn_errors))

    return ls_errors, nn_errors
