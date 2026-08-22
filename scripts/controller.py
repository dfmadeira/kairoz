from dataclasses import dataclass

import control
import matplotlib.pyplot as plt
import numpy as np
from dynamics import (
    CompositeDisturbance,
    ConstantDisturbance,
    ImpulseDisturbance,
    OUProcess,
    SineDisturbance,
    mass_spring_damper,
)
from estimation import RLS, estimate_parameters
from nn import OnlineResidualLearner
from plotting import plot_closed_loop_results
from sensor.kalman import KalmanFilter
from sensor.sensor import *
from sensor.sensor_config import *
from simulation import simulate_closed_loop, step_dynamics

from config import config as cfg

# ============================================================================
# STATE
# ============================================================================


@dataclass
class State:
    position: float
    velocity: float
    acceleration: float


# ============================================================================
# EXCITATION
# ============================================================================


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
            f0 = 0.1
            f1 = 5.0
            T = 20.0

            f = f0 + (f1 - f0) * (t / T)

            return 2.0 * np.sin(2 * np.pi * f * t)

        else:
            raise ValueError(f"Unknown excitation type: {self.kind}")


# ============================================================================
# LQR
# ============================================================================


def compute_lqr_gain(theta, Q, R, res_hat):

    theta1, theta2, theta3 = theta
    # theta = [1/m, -c/m, -k/m]

    A = np.array(
        [
            [0.0, 1.0],
            [theta3, theta2],
        ]
    )

    B = np.array(
        [
            [0.0],
            [theta1],
        ]
    )

    B_pinv = np.linalg.pinv(B)

    r_hat = np.array(
        [
            [0.0],
            [res_hat],
        ]
    )

    u_MLP = (-(B_pinv @ r_hat)).item()

    Q = np.diag(Q)

    K, _, _ = control.lqr(A, B, Q, R)

    return np.array(K).flatten(), u_MLP


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_lqr_experiment(
    m=cfg.MASS,
    c=cfg.DAMPING,
    k=cfg.SPRING,
):

    # ------------------------------------------------------------------------
    # Simulation configuration
    # ------------------------------------------------------------------------

    dt = cfg.DT
    T = cfg.SIM_TIME
    n_steps = int(T / dt)

    # ------------------------------------------------------------------------
    # TRUE PLANT STATE
    #
    # These variables represent the actual simulated physical system.
    # Kairoz algorithms should NOT directly use them.
    # ------------------------------------------------------------------------

    x = cfg.X0
    v = cfg.V0

    # ------------------------------------------------------------------------
    # Excitation
    # ------------------------------------------------------------------------

    t_switch = cfg.EXCITATION_SWITCH_TIME

    excitation = Excitation(
        cfg.EXCITATION_TYPE,
        dt,
    )

    # ------------------------------------------------------------------------
    # RLS
    # ------------------------------------------------------------------------

    estimator = RLS(lam=cfg.RLS_LAMBDA)

    theta_hat = np.zeros(3)

    # ------------------------------------------------------------------------
    # KF
    # ------------------------------------------------------------------------

    kf = KalmanFilter(dt)

    # Initial model for KF.
    #
    # This will be replaced as soon as RLS produces its first estimate.
    kf.update_model(theta_hat)

    # ------------------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------------------

    LQR_active = False
    RES_Controller = 1

    K = None

    # ------------------------------------------------------------------------
    # MLP residual learner
    # ------------------------------------------------------------------------

    learner = OnlineResidualLearner()

    residual_hat = 0.0

    # ------------------------------------------------------------------------
    # Logs
    #
    # Truth
    # ------------------------------------------------------------------------

    xs = []
    vs = []
    us = []

    # ------------------------------------------------------------------------
    # Sensor measurements
    # ------------------------------------------------------------------------

    xs_GPS = []
    xs_Encoder = []
    as_acc = []

    # ------------------------------------------------------------------------
    # KF estimates
    # ------------------------------------------------------------------------

    xs_kf = []
    vs_kf = []

    # ------------------------------------------------------------------------
    # Other logs
    # ------------------------------------------------------------------------

    disturbances = []
    residuals = []

    predicted_residuals = []
    losses = []

    theta_history = []

    # ------------------------------------------------------------------------
    # Disturbance model
    # ------------------------------------------------------------------------

    disturbance_model = CompositeDisturbance(
        [
            ConstantDisturbance(F0=cfg.CONSTANT_FORCE * int(cfg.CONSTANT_DISTURBANCE)),
            SineDisturbance(A=cfg.SINE_AMPLITUDE * cfg.SINE_ENABLE, w=cfg.SINE_FREQUENCY),
            OUProcess(
                theta=cfg.OU_THETA * cfg.OU_ENABLE, sigma=cfg.OU_SIGMA * cfg.OU_ENABLE, dt=dt
            ),
            ImpulseDisturbance(
                t0=cfg.IMPULSE_TIME,
                duration=cfg.IMPULSE_DURATION,
                A=cfg.IMPULSE_FORCE * cfg.IMPULSE_ENABLE,
            ),
        ]
        * int(cfg.ENABLE_DISTURBANCES)
    )

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    for i in range(n_steps):
        t = i * dt

        # =====================================================================
        # 1. CONTROL INPUT
        #
        # IMPORTANT:
        # The controller uses the PREVIOUS KF estimate.
        # It does NOT get the true x and v.
        # =====================================================================

        if t < t_switch:
            u = excitation(t)

        elif t == t_switch:
            K, u_MLP = compute_lqr_gain(
                theta_hat,
                cfg.Q,
                cfg.R,
                residual_hat,
            )

            LQR_active = True
            RES_Controller = 1

            # Estimated state
            x_hat = kf.state[0, 0]
            v_hat = kf.state[1, 0]

            u = float(-K @ np.array([x_hat, v_hat])) + u_MLP * RES_Controller

            u = np.clip(
                u,
                -5.0,
                5.0,
            )

        else:
            K, u_MLP = compute_lqr_gain(
                theta_hat,
                cfg.Q,
                cfg.R,
                residual_hat,
            )

            # ---------------------------------------------------------------
            # USE KF STATE — NOT TRUE STATE
            # ---------------------------------------------------------------

            x_hat = kf.state[0, 0]
            v_hat = kf.state[1, 0]

            u = -K @ np.array([x_hat, v_hat]) + u_MLP * RES_Controller

            u = np.clip(
                u,
                -5.0,
                5.0,
            )

        # =====================================================================
        # 2. DISTURBANCE
        #
        # The disturbance acts on the TRUE physical plant.
        # =====================================================================

        disturbance = disturbance_model.force(
            t,
            x,
            v,
        )

        # =====================================================================
        # 3. TRUE PLANT
        #
        # x, v, a remain the GOD state.
        # =====================================================================

        x, v, a = step_dynamics(
            x,
            v,
            u + disturbance,
            m,
            c,
            k,
            dt,
        )

        # =====================================================================
        # 4. CREATE TRUE STATE OBJECT
        # =====================================================================

        state = State(
            position=x,
            velocity=v,
            acceleration=a,
        )

        # =====================================================================
        # 5. SENSOR MEASUREMENTS
        #
        # Sensors see the true physical state and corrupt it according
        # to their own sensor models.
        # =====================================================================

        gps = GPSSensor(GPS[1])

        encoder = EncoderSensor(ENCODERS[1])

        acc = Accelerometer(ACCELEROMETERS[1])

        x_GPS = gps.measure(state)

        x_Encoder = encoder.measure(state)

        a_acc = acc.measure(state)

        # Measurement vector:
        #
        # z =
        #
        # [ GPS position       ]
        # [ Encoder position   ]
        # [ Accelerometer      ]

        z = np.array(
            [
                x_GPS,
                x_Encoder,
                a_acc,
            ]
        )

        # Store measurements

        xs_GPS.append(x_GPS)
        xs_Encoder.append(x_Encoder)
        as_acc.append(a_acc)

        # =====================================================================
        # 6. KF
        #
        # First predict using the applied input.
        # Then correct using sensor measurements.
        # =====================================================================

        kf.predict(u)

        kf.update(
            z,
            u,
        )

        # Current estimated state

        x_hat = kf.state[0, 0]
        v_hat = kf.state[1, 0]

        # Store KF estimates

        xs_kf.append(x_hat)
        vs_kf.append(v_hat)

        # =====================================================================
        # 7. RESIDUAL LEARNING
        #
        # IMPORTANT:
        # MLP gets estimated state, NOT true state.
        #
        # Accelerometer is the measured acceleration.
        # =====================================================================

        if estimator.prev_v is None:
            residual = 0.0
            residual_hat = 0.0
            loss = 0.0

        else:
            # ---------------------------------------------------------------
            # RLS/model prediction based on estimated state
            # ---------------------------------------------------------------

            phi = np.array(
                [
                    u,
                    v_hat,
                    x_hat,
                ]
            )

            a_pred = theta_hat @ phi

            # ---------------------------------------------------------------
            # Actual measured acceleration comes from sensor
            # ---------------------------------------------------------------

            residual = a_acc - a_pred

            # ---------------------------------------------------------------
            # MLP sees estimated state
            # ---------------------------------------------------------------

            residual_hat, loss = learner.update(
                x_hat,
                v_hat,
                u,
                residual,
                t,
            )

        # =====================================================================
        # 8. STORE LOGS
        # =====================================================================

        # TRUE STATE
        xs.append(x)
        vs.append(v)

        # INPUT
        us.append(u)

        # DISTURBANCE
        disturbances.append(disturbance)

        # RESIDUAL
        residuals.append(residual)

        # MLP
        predicted_residuals.append(residual_hat)

        losses.append(loss if loss is not None else losses[-1] if losses else 0.0)

        # =====================================================================
        # 9. RLS
        #
        # IMPORTANT:
        # RLS now receives the estimated state.
        # It does NOT receive the GOD state.
        # =====================================================================

        new_theta = estimator.update(
            x_hat,
            v_hat,
            u,
        )

        if new_theta is not None:
            theta_hat = new_theta.copy()

            # Update KF's internal model using the
            # newly identified dynamics.

            kf.update_model(theta_hat)

        theta_history.append(theta_hat.copy())

        # =====================================================================
        # 10. CONTROLLER UPDATE
        # =====================================================================

        if new_theta is not None and LQR_active:
            # K can be regenerated here if desired.
            pass

    # =========================================================================
    # RETURN
    # =========================================================================

    return (
        xs,  # TRUE position
        vs,  # TRUE velocity
        us,  # applied input
        disturbances,  # TRUE disturbances
        residuals,  # residual
        predicted_residuals,  # MLP prediction
        losses,  # MLP loss
        np.array(theta_history),
        xs_GPS,  # GPS measurement
        xs_Encoder,  # encoder measurement
        xs_kf,  # KF estimated position
        # vs_kf,               # KF estimated velocity
    )
