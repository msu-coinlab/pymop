import numpy as np

from pymop.problem import Problem


class Kursawe(Problem):
    def __init__(self):
        Problem.__init__(self)
        self.n_var = 3
        self.n_constr = 0
        self.n_obj = 2
        self.func = self.evaluate_
        self.xl = -5 * np.ones(self.n_var)
        self.xu = 5 * np.ones(self.n_var)

    def evaluate_(self, x, f):
        for i in range(2):
            f[:, 0] += -10 * np.exp(-0.2 * np.sqrt(np.square(x[:, i]) + np.square(x[:, i + 1])))
        f[:, 1] += np.sum(np.power(np.abs(x), 0.8) + 5 * np.sin(np.power(x, 3)), axis=1)
