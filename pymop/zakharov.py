import numpy as np

from pymop.problem import Problem


class Zakharov(Problem):
    def __init__(self, n_var=2):
        Problem.__init__(self)
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = -10 * np.ones(self.n_var)
        self.xu = 10 * np.ones(self.n_var)

    def _evaluate(self, x, f):
        a = np.sum(0.5 * np.arange(1, self.n_var + 1) * x, axis=1)
        f[:, 0] = np.sum(np.square(x), axis=1) + np.square(a) + np.power(a, 4)


