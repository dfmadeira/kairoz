import matplotlib.pyplot as plt
import numpy as np


# def plot_results(t, x, v, u):
#     plt.figure()
#     plt.plot(t, x)
#     plt.title("x(t)")
#     plt.grid()
#     plt.savefig("x.png")

#     plt.figure()
#     plt.plot(t, v)
#     plt.title("x_dot(t)")
#     plt.grid()
#     plt.savefig("v.png")

#     plt.figure()
#     plt.plot(t, u)
#     plt.title("u(t)")
#     plt.grid()
#     plt.savefig("u.png")

def plot_position_measurements(
    xs,
    x_gps,
    x_encoder,
    x_kalman,
    save_path="position_measurements.png",
):
    """
    Plot true position against sensor measurements
    and Kalman-filtered position.
    """

    gps_error = np.asarray(xs) - np.asarray(x_gps)
    encoder_error = np.asarray(xs) - np.asarray(x_encoder)
    kf_error = np.asarray(xs) - np.asarray(x_kalman)

    print("GPS MAE:", np.mean(np.abs(gps_error)))
    print("Encoder MAE:", np.mean(np.abs(encoder_error)))
    print("KF MAE:", np.mean(np.abs(kf_error)))

    plt.figure(figsize=(10, 5))

    # Measurements (behind)
    plt.plot(
        x_gps,
        label="GPS",
        color="tab:blue",
        alpha=0.45,
        linewidth=1.5,
        zorder=1,
    )

    plt.plot(
        x_encoder,
        label="Encoder",
        color="tab:red",
        alpha=0.45,
        linewidth=1.5,
        zorder=2,
    )

    # Kalman estimate
    plt.plot(
        x_kalman,
        label="Kalman",
        color="tab:green",
        linewidth=2.0,
        zorder=3,
    )

    # Ground truth (front)
    plt.plot(
        xs,
        label="Truth",
        color="black",
        linewidth=2.5,
        zorder=4,
    )

    plt.title("Position Measurements")
    plt.xlabel("Time Step")
    plt.ylabel("Position [m]")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_kalman_position(
    xs,
    x_kalman,
    save_path="kalman_position.png",
):
    """
    Plot true position against Kalman-filtered position.
    """

    plt.figure(figsize=(10, 5))

    # Kalman estimate
    plt.plot(
        x_kalman,
        label="Kalman",
        color="tab:green",
        linewidth=3,
        zorder=1,
    )

    # Ground truth
    plt.plot(
        xs,
        label="Truth",
        color="black",
        linewidth=1,
        zorder=2,
    )

    plt.title("Kalman Position Estimate")
    plt.xlabel("Time Step")
    plt.ylabel("Position [m]")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_kalman_error(
    xs,
    x_kalman,
    save_path="kalman_position_error.png",
):
    """
    Plot Kalman position estimation error.
    """

    error = np.asarray(xs) - np.asarray(x_kalman)

    plt.figure(figsize=(10, 5))

    plt.plot(
        error,
        label="Position Error",
        linewidth=1.5,
    )

    print("Max error:", np.max(np.abs(error)))
    print("Mean absolute error:", np.mean(np.abs(error)))
    print("RMSE:", np.sqrt(np.mean(error**2)))

    plt.title("Kalman Position Estimation Error")
    plt.xlabel("Time Step")
    plt.ylabel("Error [m]")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_closed_loop_results(t, x, v, u, label="system"):

    # POSITION
    plt.figure()
    plt.plot(t, x)
    plt.title(f"x(t) - {label}")
    plt.grid()
    plt.savefig(f"x_{label}.png")

    # VELOCITY
    plt.figure()
    plt.plot(t, v)
    plt.title(f"x_dot(t) - {label}")
    plt.grid()
    plt.savefig(f"v_{label}.png")

    # CONTROL INPUT
    plt.figure()
    plt.plot(t, u)
    plt.title(f"u(t) - {label}")
    plt.grid()
    plt.savefig(f"u_{label}.png")


def plot_results1(xs, vs, us, dt=0.01, save_path="lqr_experiment.png"):

    t = np.arange(len(xs)) * dt

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t, xs)
    plt.ylabel("x")
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t, vs)
    plt.ylabel("v")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t, us)
    plt.ylabel("u")
    plt.xlabel("t")
    plt.grid()

    plt.tight_layout()

    # 🔥 critical: always save
    plt.savefig(save_path, dpi=300)

    plt.close()  # prevents memory leaks in repeated runs


def plot_results(
    xs,
    vs,
    us,
    disturbances,
    residuals,
    dt=0.01,
    save_dir=".",
):

    t = np.arange(len(xs)) * dt

    # -------------------------------------------------
    # Position
    # -------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(t, xs)
    plt.title("Position")
    plt.xlabel("Time [s]")
    plt.ylabel("x")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/position.png", dpi=300)
    plt.close()

    # -------------------------------------------------
    # Velocity
    # -------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(t, vs)
    plt.title("Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("v")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/velocity.png", dpi=300)
    plt.close()

    # -------------------------------------------------
    # Input
    # -------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(t, us)
    plt.title("Excitation Input")
    plt.xlabel("Time [s]")
    plt.ylabel("u")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/input.png", dpi=300)
    plt.close()

    # -------------------------------------------------
    # True disturbance
    # -------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(t, disturbances)
    plt.title("True Disturbance")
    plt.xlabel("Time [s]")
    plt.ylabel("d(t)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/disturbance.png", dpi=300)
    plt.close()

    # -------------------------------------------------
    # Residual
    # -------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(t, residuals)
    plt.title("Residual")
    plt.xlabel("Time [s]")
    plt.ylabel("Residual")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/residual.png", dpi=300)
    plt.close()

    # -------------------------------------------------
    # Disturbance vs Residual
    # -------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(t, disturbances, label="True Disturbance", linewidth=2)
    plt.plot(t, residuals, label="Residual", linewidth=2)
    plt.title("Residual vs True Disturbance")
    plt.xlabel("Time [s]")
    plt.ylabel("Force / Acceleration")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/disturbance_vs_residual.png", dpi=300)
    plt.close()

def plot_mlp_comparison(
    residuals,
    disturbances,
    predictions,
    windows,
    online_prediction,
    dt=0.01,
    save_path="mlp_comparison.png",
):

    # -------------------------------------------------
    # Time vector
    # -------------------------------------------------
    t = np.arange(len(residuals)) * dt

    # -------------------------------------------------
    # Figure
    # -------------------------------------------------
    plt.figure(figsize=(12, 6))

    # True disturbance
    plt.plot(
        t,
        disturbances,
        linewidth=2,
        label="True disturbance",
    )

    # Residual
    plt.plot(
        t,
        residuals,
        linewidth=2,
        label="Residual",
    )

    # Offline MLPs
    for pred, seconds in zip(predictions, windows):

        plt.plot(
            t,
            pred,
            linewidth=1.5,
            label=f"Offline MLP ({seconds}s)",
        )

    # Online MLP
    plt.plot(
        t,
        online_prediction,
        linewidth=2,
        linestyle="--",
        label="Online MLP",
    )

    # -------------------------------------------------
    # Cosmetics
    # -------------------------------------------------
    plt.title("Residual Learning Comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("Residual / Disturbance")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()

def plot_online_learning(
    residuals,
    online_prediction,
    disturbances=None,
    losses=None,
    dt=0.01,
    save_path="online_learning.png",
):

    t = np.arange(len(residuals)) * dt

    # -------------------------------------------------
    # Residual prediction
    # -------------------------------------------------
    plt.figure(figsize=(12, 6))

    plt.plot(
        t,
        residuals,
        linewidth=2,
        label="Residual",
    )

    plt.plot(
        t,
        online_prediction,
        linewidth=1,
        label="Online MLP",
    )

    if disturbances is not None:

        plt.plot(
            t,
            disturbances,
            linewidth=2,
            alpha=0.7,
            label="True disturbance",
        )

    plt.title("Online Residual Learning")

    plt.xlabel("Time [s]")
    plt.ylabel("Residual")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()

    # -------------------------------------------------
    # Training loss
    # -------------------------------------------------
    if losses is not None:

        plt.figure(figsize=(12,4))

        plt.plot(
            t,
            losses,
            linewidth=2,
        )

        plt.title("Online MLP Training Loss")

        plt.xlabel("Time [s]")

        plt.ylabel("MSE")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            "online_loss.png",
            dpi=300,
        )

        plt.close()
        def plot_theta_history(
            theta_history,
            theta_true,
            dt=0.01,
            save_path="theta_history.png",
        ):

            t = np.arange(theta_history.shape[0]) * dt

            labels = [
                r"$1/m$",
                r"$-c/m$",
                r"$-k/m$",
            ]

            plt.figure(figsize=(12,8))

            for i in range(3):

                plt.subplot(3,1,i+1)

                plt.plot(
                    t,
                    theta_history[:,i],
                    label="Estimated",
                    linewidth=2,
                )

                plt.axhline(
                    theta_true[i],
                    color="red",
                    linestyle="--",
                    label="True",
                )

                plt.ylabel(labels[i])

                plt.grid(True)

                if i == 0:
                    plt.legend()

            plt.xlabel("Time [s]")

            plt.tight_layout()

            plt.savefig(
                save_path,
                dpi=300,
            )

            plt.close()

def plot_theta_history(
    theta_history,
    theta_true,
    dt=0.01,
    save_path="theta_history.png",
):

    t = np.arange(theta_history.shape[0]) * dt

    labels = [
        r"$1/m$",
        r"$-c/m$",
        r"$-k/m$",
    ]

    plt.figure(figsize=(12,8))

    for i in range(3):

        plt.subplot(3,1,i+1)

        plt.plot(
            t,
            theta_history[:,i],
            label="Estimated",
            linewidth=2,
        )

        plt.axhline(
            theta_true[i],
            color="red",
            linestyle="--",
            label="True",
        )

        plt.ylabel(labels[i])

        plt.grid(True)

        if i == 0:
            plt.legend()

    plt.xlabel("Time [s]")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()

def plot_theta_error(
    theta_history,
    theta_true,
    dt=0.01,
    save_path="theta_error.png",
):

    theta_history = np.asarray(theta_history)

    t = np.arange(theta_history.shape[0]) * dt

    # -------------------------------------------------
    # Parameter estimation error
    # -------------------------------------------------
    theta_error = theta_history - np.asarray(theta_true)

    labels = [
        r"$1/m$ error",
        r"$-c/m$ error",
        r"$-k/m$ error",
    ]

    plt.figure(figsize=(12, 8))

    for i in range(3):

        plt.subplot(3, 1, i + 1)

        plt.plot(
            t,
            theta_error[:, i],
            linewidth=2,
        )

        plt.axhline(
            0.0,
            color="black",
            linestyle="--",
        )

        plt.ylabel(labels[i])

        plt.grid(True)

    plt.xlabel("Time [s]")

    plt.suptitle("RLS Parameter Estimation Error")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()
