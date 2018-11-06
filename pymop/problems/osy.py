import numpy as np

from pymop import load_pareto_front_from_file
from pymop.problem import Problem


class OSY(Problem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=2, n_constr=6, type_var=np.double)
        self.xl = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        self.xu = np.array([10.0, 10.0, 5.0, 6.0, 5.0, 10.0])

    def _evaluate(self, x, f, g, *args, **kwargs):
        f[:, 0] = -(25 * np.square(x[:, 0] - 2) + np.square(x[:, 1] - 2) + np.square(x[:, 2] - 1) \
                    + np.square(x[:, 3] - 4) + + np.square(x[:, 4] - 1))

        f[:, 1] = np.sum(np.square(x), axis=1)

        g[:, 0] = (x[:, 0] + x[:, 1] - 2.0) / 2.0
        g[:, 1] = (6.0 - x[:, 0] - x[:, 1]) / 6.0
        g[:, 2] = (2.0 - x[:, 1] + x[:, 0]) / 2.0
        g[:, 3] = (2.0 - x[:, 0] + 3.0 * x[:, 1]) / 2.0
        g[:, 4] = (4.0 - np.square(x[:, 2] - 3.0) - x[:, 3]) / 4.0
        g[:, 5] = (np.square(x[:, 4] - 3.0) + x[:, 5] - 4.0) / 4.0
        g[:, :] = - g[:, :]

    def _calc_pareto_front(self):
        return load_pareto_front_from_file("osy.pf")
