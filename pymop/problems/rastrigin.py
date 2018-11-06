import numpy as np

from pymop.problem import Problem


class Rastrigin(Problem):
    def __init__(self, n_var=2, A=10.0):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-5, xu=5, type_var=np.double)
        self.A = A

    def _evaluate(self, x, f, *args, **kwargs):
        z = np.power(x, 2) - self.A * np.cos(2 * np.pi * x)
        f[:, 0] = self.A * self.n_var + np.sum(z, axis=1)
