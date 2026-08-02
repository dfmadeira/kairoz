from controller import Excitation, run_lqr_experiment
from dynamics import mass_spring_damper
from plotting import plot_results, plot_results1, plot_mlp_comparison, plot_online_learning, plot_theta_error
from nn import train_residual_mlp, predict_residual, OnlineResidualLearner
import numpy as np

# if __name__ == "__main__":
#     # t, x, v, u = mass_spring_damper(u_func=Excitation("prbs"))
#     # plot_results(t, x, v, u)
#     xs, vs, us, disturbances, residuals = run_lqr_experiment()
#     plot_results(xs, vs, us, disturbances, residuals)


if __name__ == "__main__":

    # -------------------------------------------------
    # Generate simulation
    # -------------------------------------------------
    xs, vs, us, disturbances, residuals, online_prediction, losses, theta_history = run_lqr_experiment()

    plot_results(
        xs,
        vs,
        us,
        disturbances,
        residuals,
    )

    # # -------------------------------------------------
    # # Build dataset
    # # -------------------------------------------------
    # X = np.column_stack((xs, vs, us))
    # Y = np.array(residuals).reshape(-1, 1)

    # # -------------------------------------------------
    # # Train three MLPs
    # # -------------------------------------------------
    # windows = [10, 20, 30]

    # models = []
    # predictions = []

    # for seconds in windows:

    #     model = train_residual_mlp(
    #         X,
    #         Y,
    #         train_seconds=seconds,
    #         dt=0.01,
    #     )

    #     y_hat = predict_residual(
    #         model,
    #         X,
    #     )

    #     models.append(model)
    #     predictions.append(y_hat)


    # # -------------------------------------------------
    # # Compare predictions
    # # -------------------------------------------------
    # plot_mlp_comparison(
    #     residuals=residuals,
    #     disturbances=disturbances,
    #     predictions=predictions,
    #     windows=windows,
    #     online_prediction=online_prediction,
    # )

    plot_online_learning(
        residuals,
        online_prediction,
        disturbances,
        losses,
    )

    theta_true = np.array([1, 0.4, 2])

    plot_theta_error(
        theta_history,
        theta_true
    )
