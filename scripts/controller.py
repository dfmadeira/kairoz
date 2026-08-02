import control
import matplotlib.pyplot as plt
import numpy as np
from config import *
from dynamics import (
    CompositeDisturbance,
    ConstantDisturbance,
    ImpulseDisturbance,
    OUProcess,
    SineDisturbance,
    mass_spring_damper,
)
from estimation import RLS, estimate_parameters
from plotting import plot_closed_loop_results
from simulation import simulate_closed_loop, step_dynamics
from nn import OnlineResidualLearner


class Excitation:
    def __init__(self, kind="prbs", dt=0.01):
        self.kind = kind
        self.dt = dt
        self.t0 = 0

    def __call__(self, t):
        if self.kind == "prbs":
            return np.random.choice([-5.0, 5.0])

        elif self.kind == "sine":
            return 30.0 * np.sin(2 * np.pi * 0.5 * t)

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


def compute_lqr_gain(theta, Q, R, res_hat):
    theta1, theta2, theta3 = theta  # = [1/m, -c/m, -k/m]

    A = np.array([[0, 1], [theta3, theta2]])

    B = np.array([[0], [theta1]])

    B_pinv = np.linalg.pinv(B)

    r_hat = np.array([[0.0],
                    [res_hat]])

    u_MLP = float(-(B_pinv @ r_hat))

    Q = np.diag(Q)

    K, _, _ = control.lqr(A, B, Q, R)

    return np.array(K).flatten(), u_MLP #if ENABLE_LQR else 0


def run_lqr_experiment(m=MASS, c=DAMPING, k=SPRING):

    dt = DT
    T = SIM_TIME
    n_steps = int(T / dt)

    x = X0
    v = V0

    t_switch = EXCITATION_SWITCH_TIME

    excitation = Excitation(EXCITATION_TYPE)

    estimator = RLS(lam=RLS_LAMBDA)

    LQR_active = False

    xs, vs, us = [], [], []
    disturbances = []
    residuals = []

    theta_history = []

    theta_hat = np.zeros(3)

    K = None

    learner = OnlineResidualLearner()

    predicted_residuals = []
    losses = []

    disturbance_model = CompositeDisturbance(
        [
            ConstantDisturbance(F0=CONSTANT_FORCE * int(CONSTANT_DISTURBANCE)) ,
            SineDisturbance(A=SINE_AMPLITUDE * SINE_ENABLE, w=SINE_FREQUENCY),
            OUProcess(theta=OU_THETA * OU_ENABLE, sigma=OU_SIGMA * OU_ENABLE, dt=dt),
            ImpulseDisturbance(t0=IMPULSE_TIME, duration=IMPULSE_DURATION, A=IMPULSE_FORCE * IMPULSE_ENABLE),
        ] * int(ENABLE_DISTURBANCES)
    )
    RES_Controller = 1

    for i in range(n_steps):

        t = i * dt

        # -------------------------------------------------
        # 1. Generate input
        # -------------------------------------------------
        #
        if t < t_switch:
            u = excitation(t)

        elif t == t_switch:
            # print(theta_hat)

            K, u_MLP = compute_lqr_gain(theta_hat, Q, R, residual_hat)

            LQR_active = True
            RES_Controller = 1
            u = float(-K @ np.array([x, v])) + u_MLP * RES_Controller
            u = np.clip(u, -5.0, 5.0)

        else:
            K, u_MLP = compute_lqr_gain(theta_hat, Q, R, residual_hat)

            u = -K @ np.array([x, v]) + u_MLP * RES_Controller #LQR innactive
            u = np.clip(u, -5.0, 5.0)
        # -------------------------------------------------
        # 2. Disturbance
        # -------------------------------------------------
        disturbance = disturbance_model.force(t, x, v)

        # -------------------------------------------------
        # 3. Simulate plant
        # -------------------------------------------------
        x, v = step_dynamics(
            x,
            v,
            u + disturbance,
            m,
            c,
            k,
            dt,
        )

        # -------------------------------------------------
        # 4. Measured acceleration
        # -------------------------------------------------
        if estimator.prev_v is None:
            a_measured = 0.0
            residual = 0.0

            residual_hat = 0.0
            loss = 0.0

        else:
            a_measured = (v - estimator.prev_v) / dt

            # Prediction using PREVIOUS theta
            phi = np.array([u, v, x])

            a_pred = theta_hat @ phi

            residual = a_measured - a_pred

            residual_hat, loss = learner.update(
                x,
                v,
                u,
                residual,
                t,
            )

        # -------------------------------------------------
        # 5. Store logs
        # -------------------------------------------------
        xs.append(x)
        vs.append(v)
        us.append(u)

        disturbances.append(disturbance)
        residuals.append(residual)

        predicted_residuals.append(residual_hat)
        losses.append(loss if loss is not None else losses[-1] if losses else 0.0)

        # -------------------------------------------------
        # 6. Update RLS
        # -------------------------------------------------
        new_theta = estimator.update(x, v, u)

        if new_theta is not None:
            theta_hat = new_theta.copy()

        theta_history.append(theta_hat.copy())

        # -------------------------------------------------
        # 7. Update controller
        # -------------------------------------------------
        if (new_theta is not None) and LQR_active:
            # K = compute_lqr_gain(theta_hat, Q, R)
            # print(K)
            p = 1

    return (
        xs,
        vs,
        us,
        disturbances,
        residuals,
        predicted_residuals,
        losses,
        np.array(theta_history)
    )
