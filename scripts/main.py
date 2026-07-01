from controller import Excitation, run_lqr_experiment
from dynamics import mass_spring_damper
from plotting import plot_results

if __name__ == "__main__":
    t, x, v, u = mass_spring_damper(u_func=Excitation("sine"), c=-0.2)
    plot_results(t, x, v, u)
    run_lqr_experiment(c=-0.2)
