import numpy as np

from pymop.problem import Problem


class Rosenbrock(Problem):
    def __init__(self, n_var=2, **kwargs):
        Problem.__init__(self, **kwargs)
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = 1
        self.func = self.evaluate_
        self.xl = -2.048 * np.ones(self.n_var)
        self.xu = 2.048 * np.ones(self.n_var)

    def evaluate_(self, x, f):
        for i in range(x.shape[1] - 1):
            f[:, 0] += 100 * np.square((x[:, i + 1] - np.square(x[:, i]))) + np.square((1 - x[:, i]))

            #double temp1 = (x[i] * x[i]) - x[i + 1];
            #double temp2 = x[i] - 1.0;
            #sum += (100.0 * temp1 * temp1) + (temp2 * temp2);