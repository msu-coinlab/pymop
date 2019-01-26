import numpy as np


def plot_problem_surface(problem, n_samples, plot_type="wireframe"):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except:
        raise Exception("Please install 'matplotlib' to use the plotting functionality.")

    if problem.n_var == 1 and problem.n_obj == 1:

        X = np.linspace(problem.xl[0], problem.xu[0], num=n_samples)[:, None]
        Y = problem.evaluate(X, return_values_of=["F"])
        plt.plot(X, Y)

    elif problem.n_var == 2 and problem.n_obj == 1:


        X_range = np.linspace(problem.xl[0], problem.xu[0], num=n_samples)
        Y_range = np.linspace(problem.xl[1], problem.xu[1], num=n_samples)
        X, Y = np.meshgrid(X_range, Y_range)

        A = np.zeros((n_samples * n_samples, 2))
        counter = 0
        for i, x in enumerate(X_range):
            for j, y in enumerate(Y_range):
                A[counter, 0] = x
                A[counter, 1] = y
                counter += 1

        F = np.reshape(problem.evaluate(A, return_values_of=["F"]), (n_samples, n_samples))

        fig = plt.figure()
        # Plot the surface.

        if plot_type == "wireframe":
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(X, Y, F)
        elif plot_type == "contour":
            CS = plt.contour(X, Y, F)
            plt.clabel(CS, inline=1, fontsize=10)
        else:
            raise Exception("Unknown plotting method.")


    else:
        raise Exception("Can only plot problems with less than two variables and one objective.")

    plt.show()
