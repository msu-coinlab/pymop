import numpy as np

from pymop import load_pareto_front_from_file
from pymop.problem import Problem


class Kursawe(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=-5, xu=5, type_var=np.double)

    def _evaluate(self, x, f, *args, **kwargs):
        for i in range(2):
            f[:, 0] += -10 * np.exp(-0.2 * np.sqrt(np.square(x[:, i]) + np.square(x[:, i + 1])))
        f[:, 1] += np.sum(np.power(np.abs(x), 0.8) + 5 * np.sin(np.power(x, 3)), axis=1)

    def _calc_pareto_front(self):
        return load_pareto_front_from_file("kursawe.pf")
