import numpy as np

from pymop.problem import Problem


class Truss2D(Problem):

    def __init__(self):
        Problem.__init__(self)
        self.n_var = 3
        self.n_constr = 1
        self.n_obj = 2
        self.func = self._evaluate

        self.Amax = 0.01
        self.Smax = 1e5

        self.xl = np.array([0.0, 0.0, 1.0])
        self.xu = np.array([self.Amax, self.Amax, 3.0])

    def _evaluate(self, x, f, g):

        # variable names for convenient access
        x1 = x[:, 0]
        x2 = x[:, 1]
        y = x[:, 2]

        # first objectives
        f[:, 0] = x1 * np.sqrt(16 + np.square(y)) + x2 * np.sqrt((1 + np.square(y)))

        # measure which are needed for the second objective
        sigma_ac = 20 * np.sqrt(16 + np.square(y)) / (y * x1)
        sigma_bc = 80 * np.sqrt(1 + np.square(y)) / (y * x2)

        # take the max
        f[:, 1] = np.max(np.column_stack((sigma_ac, sigma_bc)), axis=1)

        # define a constraint
        g[:, 0] = f[:, 1] - self.Smax
