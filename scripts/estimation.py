import numpy as np


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
