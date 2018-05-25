import numpy as np

from pymop.problem import Problem


class BNH(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)
        self.n_var = 2
        self.n_constr = 2
        self.n_obj = 2
        self.func = self.evaluate_
        self.xl = np.zeros(self.n_var)
        self.xu = np.array([5.0, 3.0])

    def evaluate_(self, x, f, g):
        f[:, 0] = 4 * x[:, 0] ** 2 + 4 * x[:, 1] ** 2
        f[:, 1] = (x[:, 0] - 5) ** 2 + (x[:, 1] - 5) ** 2
        g[:, 0] = (1 / 25) * ((x[:, 0] - 5) ** 2 + x[:, 1] ** 2 - 25)
        g[:, 1] = -1 / 7.7 * ((x[:, 0] - 8) ** 2 + (x[:, 1] + 3) ** 2 - 7.7)
