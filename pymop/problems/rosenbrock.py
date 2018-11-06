import numpy as np

from pymop.problem import Problem


class Rosenbrock(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-2.048, xu=2.048, type_var=np.double)

    def _evaluate(self, x, f, *args, **kwargs):
        for i in range(x.shape[1] - 1):
            f[:, 0] += 100 * np.square((x[:, i + 1] - np.square(x[:, i]))) + np.square((1 - x[:, i]))
