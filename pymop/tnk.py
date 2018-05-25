import numpy as np

from pymop.problem import Problem


class TNK(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)
        self.n_var = 2
        self.n_constr = 2
        self.n_obj = 2
        self.func = self.evaluate_
        self.xl = np.array([0, 0])
        self.xu = np.array([np.pi, np.pi])

    def evaluate_(self, x, f, g):
        f[:, 0] = x[:, 0]
        f[:, 1] = x[:, 1]
        g[:, 0] = -(np.square(x[:, 0]) + np.square(x[:, 1]) - 1.0 - 0.1 * np.cos(16.0 * np.arctan(x[:, 0] / x[:, 1])))
        g[:, 1] = 2 * (np.square(x[:, 0] - 0.5) + np.square(x[:, 1] - 0.5)) - 1
