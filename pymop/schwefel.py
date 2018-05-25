import numpy as np

from pymop.problem import Problem


class Schwefel(Problem):
    def __init__(self, n_var=2, **kwargs):
        Problem.__init__(self, **kwargs)
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = 1
        self.func = self.evaluate_
        self.xl = -500 * np.ones(self.n_var)
        self.xu = 500 * np.ones(self.n_var)

    def evaluate_(self, x, f):
        f[:, 0] = 418.9829 * self.n_var - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
