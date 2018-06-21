import numpy as np

from pymop.problem import Problem


class Rosenbrock(Problem):
    def __init__(self, n_var=2):
        Problem.__init__(self)
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = -2.048 * np.ones(self.n_var)
        self.xu = 2.048 * np.ones(self.n_var)

    def _evaluate(self, x, f):
        for i in range(x.shape[1] - 1):
            f[:, 0] += 100 * np.square((x[:, i + 1] - np.square(x[:, i]))) + np.square((1 - x[:, i]))