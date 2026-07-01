import control
import matplotlib.pyplot as plt
import numpy as np

from dynamics import mass_spring_damper
from estimation import estimate_parameters
from plotting import plot_closed_loop_results
from simulation import simulate_closed_loop


class Excitation:
    def __init__(self, kind="prbs", dt=0.01):
        self.kind = kind
        self.dt = dt
        self.t0 = 0

    def __call__(self, t):
        if self.kind == "prbs":
            return np.random.choice([-2.0, 2.0])

        elif self.kind == "sine":
            return 2.0 * np.sin(2 * np.pi * 0.5 * t)

        elif self.kind == "step":
            return 2.0 if t > 5 else 0.0

        elif self.kind == "noise":
            return np.random.randn()

        elif self.kind == "chirp":
            # frequency increases over time
            f0 = 0.1
            f1 = 5.0
            T = 20.0

            f = f0 + (f1 - f0) * (t / T)
            return 2.0 * np.sin(2 * np.pi * f * t)

        else:
            raise ValueError(f"Unknown excitation type: {self.kind}")


def compute_lqr_gain(theta):
    theta1, theta2, theta3 = theta  # = [1/m, -c/m, -k/m]

    A = np.array([[0, 1], [theta3, theta2]])

    B = np.array([[0], [theta1]])

    Q = np.diag([10, 1])  # tune this
    R = np.array([[1]])

    K, _, _ = control.lqr(A, B, Q, R)

    return np.array(K).flatten()


def run_lqr_experiment(
    m=1,
    c=0.4,
    k=2,
):

    # open-loop dataset for identification
    t, x, v, u = mass_spring_damper(m=m, c=c, k=k, u_func=Excitation("prbs"))

    theta_true = np.array([1 / m, -c / m, -k / m])

    theta_ls = estimate_parameters(t, x, v, u, dt=0.01)

    print(theta_ls)

    # closed-loop simulations
    #
    K = compute_lqr_gain(theta_ls)

    t1, x1, v1, u1 = simulate_closed_loop(K, m, c, k)

    # plots
    plot_closed_loop_results(t1, x1, v1, u1, label="LS_LQR")

    # comparison plot (position only for clarity)
    plt.figure()
    plt.plot(t1, x1, label="LS-LQR")
    plt.legend()
    plt.title("Closed-loop position comparison")
    plt.grid()
    plt.savefig("comparison_x.png")

    # errors
    print("LS θ error:", np.linalg.norm(theta_true - theta_ls))
