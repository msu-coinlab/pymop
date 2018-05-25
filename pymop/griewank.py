import numpy as np

from pymop.problem import Problem


class Griewank(Problem):
    def __init__(self,n_var=2, **kwargs):
        Problem.__init__(self, **kwargs)
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = 1
        self.func = self.evaluate_
        self.xl = -600 * np.ones(self.n_var)
        self.xu = 600 * np.ones(self.n_var)

    def evaluate_(self, x, f):
        f[:,0] = 1 + 1 / 4000 * np.sum(np.power(x,2), axis=1) \
                                 - np.prod(np.cos(x / np.sqrt(np.arange(1,x.shape[1]+1))), axis=1)
