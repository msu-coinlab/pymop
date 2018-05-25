import numpy as np

from pymop.problem import Problem


class Sphere(Problem):
    def __init__(self, n_var=10):
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = 1
        self.func = self.evaluate_
        self.xl = 0 * np.ones(self.n_var)
        self.xu = 1 * np.ones(self.n_var)

    def evaluate_(self, x, f):
        f[:, 0] = np.sum(np.square(x - 0.5), axis=1)
