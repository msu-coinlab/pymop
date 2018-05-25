import numpy as np

from pymop.problem import Problem


class WeldedBeam(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)
        self.n_var = 4
        self.n_constr = 4
        self.n_obj = 2
        self.func = self.evaluate_
        self.xl = np.array([0.125, 0.1, 0.1, 0.125])
        self.xu = np.array([5.0, 10.0, 10.0, 5.0])

    def evaluate_(self, x, f, g):
        f[:, 0] = 1.10471 * x[:, 0] ** 2 * x[:, 1] + 0.04811 * x[:, 2] * x[:, 3] * (14.0 + x[:, 1])
        f[:, 1] = 2.1952 / (x[:, 3] * x[:, 2] ** 3)

        P = 6000
        L = 14
        t_max = 13600
        s_max = 30000

        R = np.sqrt(0.25 * (x[:, 1] ** 2 + (x[:, 0] + x[:, 2]) ** 2))
        M = P * (L + x[:, 1] / 2)
        J = 2 * np.sqrt(0.5) * x[:, 0] * x[:, 1] * (x[:, 1] ** 2 / 12 + 0.25 * (x[:, 0] + x[:, 2]) ** 2)
        t1 = P / (np.sqrt(2) * x[:, 0] * x[:, 1])
        t2 = M * R / J
        t = np.sqrt(t1 ** 2 + t2 ** 2 + t1 * t2 * x[:, 1] / R)
        s = 6 * P * L / (x[:, 3] * x[:, 2] ** 2)
        P_c = 64746.022 * (1 - 0.0282346 * x[:, 2]) * x[:, 2] * x[:, 3] ** 3

        g[:, 0] = (1 / t_max) * (t - t_max)
        g[:, 1] = (1 / s_max) * (s - s_max)
        g[:, 2] = (1 / (5 - 0.125)) * (x[:, 0] - x[:, 3])
        g[:, 3] = (1 / P) * (P - P_c)
