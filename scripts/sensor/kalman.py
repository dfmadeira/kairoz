import numpy as np
from .kalman_cfg import Q, R


class KalmanFilter:

    def __init__(
        self,
        dt,
        x0=None,
        P0=None,
    ):

        self.dt = dt

        # Model matrices
        self.A = None
        self.B = None
        self.H = None
        self.D = None

        # Noise matrices
        self.Q = Q
        self.R = R

        # Mass-spring state:
        # x = [position, velocity]
        n = 2

        self.x = (
            np.zeros((n, 1))
            if x0 is None
            else np.asarray(x0).reshape(n, 1)
        )

        self.P = (
            np.eye(n)
            if P0 is None
            else np.asarray(P0)
        )

    def update_model(self, theta):

        theta1, theta2, theta3 = theta

        # State transition matrix
        self.A = np.array([
            [1.0, self.dt],
            [
                theta3 * self.dt,
                1.0 + theta2 * self.dt
            ],
        ])

        # Input matrix
        self.B = np.array([
            [0.0],
            [theta1 * self.dt],
        ])

        # Measurement matrix
        self.H = np.array([
            [1.0, 0.0],      # GPS
            [1.0, 0.0],      # Encoder
            [theta3, theta2] # Accelerometer
        ])

        # Direct input feedthrough
        self.D = np.array([
            [0.0],            # GPS
            [0.0],            # Encoder
            [theta1],         # Accelerometer
        ])

    def predict(self, u):

        u = np.asarray(u).reshape(-1, 1)

        self.x = self.A @ self.x + self.B @ u

        self.P = (
            self.A @ self.P @ self.A.T
            + self.Q
        )

    def update(self, z, u=None):

        z = np.asarray(z).reshape(-1, 1)

        if u is None:
            u = np.zeros((self.B.shape[1], 1))
        else:
            u = np.asarray(u).reshape(-1, 1)

        # Predicted measurement
        z_hat = self.H @ self.x + self.D @ u

        # Innovation
        innovation = z - z_hat

        # Innovation covariance
        S = (
            self.H @ self.P @ self.H.T
            + self.R
        )

        # Kalman gain
        K = (
            self.P
            @ self.H.T
            @ np.linalg.inv(S)
        )

        # State update
        self.x = self.x + K @ innovation

        # Covariance update
        I = np.eye(self.P.shape[0])

        self.P = (
            (I - K @ self.H)
            @ self.P
        )

    @property
    def state(self):
        return self.x.copy()

    @property
    def covariance(self):
        return self.P.copy()
