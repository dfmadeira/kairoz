import matplotlib.pyplot as plt


def plot_results(t, x, v, u):
    plt.figure()
    plt.plot(t, x)
    plt.title("x(t)")
    plt.grid()
    plt.savefig("x.png")

    plt.figure()
    plt.plot(t, v)
    plt.title("x_dot(t)")
    plt.grid()
    plt.savefig("v.png")

    plt.figure()
    plt.plot(t, u)
    plt.title("u(t)")
    plt.grid()
    plt.savefig("u.png")


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
