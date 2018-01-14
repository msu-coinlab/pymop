import numpy as np

from pyop.problem import Problem


class DTLZ(Problem):
    def __init__(self, n_var, n_obj):
        Problem.__init__(self)
        self.n_obj = n_obj
        self.n_var = n_var
        self.n_constr = 0
        self.func = self.evaluate_
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

        self.k = self.n_var - self.n_obj + 1


class DTLZ1(DTLZ):
    def __init__(self, n_var, n_obj):
        super().__init__(n_var, n_obj)

    def calc_pareto_front(self):
        x1 = np.arange(0, 0.5, 100)
        return np.array([x1, 0.5 - x1]).T

    def evaluate_(self, x, f):
        g = 100 * \
            (self.k + np.sum(np.square(x[:, -self.k:] - 0.5) - np.cos(20 * np.pi * (x[:, -self.k:] - 0.5)), axis=1))
        for i in range(0, self.n_obj):
            f[:, i] = 0.5 * (1 + g)
            f[:, i] *= np.prod(x[:, :self.n_obj - 1 - i], axis=1)
            if i > 0:
                f[:, i] *= 1 - x[:, self.n_obj - 1 - i]


class DTLZ2(DTLZ):
    def __init__(self, n_var, n_obj):
        super().__init__(n_var, n_obj)

    def calc_pareto_front(self):
        pass

    def evaluate_(self, x, f):
        g = np.sum(np.square(x[:, -self.k:] - 0.5), axis=1)
        for i in range(0, self.n_obj):
            f[:, i] = (1 + g)
            f[:, i] *= np.prod(np.cos(x[:, :self.n_obj - 1 - i] * np.pi / 2.0), axis=1)
            if i > 0:
                f[:, i] *= np.sin(x[:, self.n_obj - 1 - i] * np.pi / 2.0)


class DTLZ3(DTLZ):
    def __init__(self, n_var, n_obj):
        super().__init__(n_var, n_obj)

    def calc_pareto_front(self):
        pass

    def evaluate_(self, x, f):
        g = 100 * (
            self.k + np.sum(np.square(x[:, -self.k:] - 0.5) - np.cos(20 * np.pi * (x[:, -self.k:] - 0.5)), axis=1))
        for i in range(0, self.n_obj):
            f[:, i] = (1 + g)
            f[:, i] *= np.prod(np.cos(x[:, :self.n_obj - 1 - i] * np.pi / 2.0), axis=1)
            if i > 0:
                f[:, i] *= np.sin(x[:, self.n_obj - 1 - i] * np.pi / 2.0)


class DTLZ4(DTLZ):
    def __init__(self, n_var, n_obj, alpha=100, d=100):
        super().__init__(n_var, n_obj)
        self.alpha = alpha
        self.d = d

    def calc_pareto_front(self):
        pass

    def evaluate_(self, x, f):
        g = np.sum(np.square(x[:, -self.k:] - 0.5), axis=1)
        for i in range(0, self.n_obj):
            f[:, i] = (1 + g)
            f[:, i] *= np.prod(np.cos(np.power(x[:, :self.n_obj - 1 - i], self.alpha) * np.pi / 2.0), axis=1)
            if i > 0:
                f[:, i] *= np.sin(np.power(x[:, self.n_obj - 1 - i], self.alpha) * np.pi / 2.0)


class DTLZ7(DTLZ):
    def __init__(self, n_var, n_obj):
        super().__init__(n_var, n_obj)

    def calc_pareto_front(self):
        pass

    def evaluate_(self, x, f):
        for i in range(0, self.n_obj - 1):
            f[:, i] = x[:, i]

        g = 1 + 9 / self.k * np.sum(x[:, -self.k:], axis=1)
        h = self.n_obj - np.sum(f[:, :-1] / (1 + g[:, None]) * (1 + np.sin(3 * np.pi * f[:, :-1])), axis=1)
        f[:, self.n_obj - 1] = (1 + g) * h
