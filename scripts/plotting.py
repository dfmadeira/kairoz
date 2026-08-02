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
