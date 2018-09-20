import numpy as np

from pymop.problem import Problem
from pymop.util import get_uniform_weights


class DTLZ(Problem):
    def __init__(self, n_var, n_obj, n_pareto_points=None):
        Problem.__init__(self)
        self.n_obj = n_obj
        self.n_var = n_var
        self.n_constr = 0
        self.func = self._evaluate
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

        self.k = self.n_var - self.n_obj + 1
        self.n_pareto_points = n_pareto_points

    def g1(self, X_M):
        return 100 * (self.k + np.sum(np.square(X_M - 0.5) - np.cos(20 * np.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return np.sum(np.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, f, alpha=1):
        for i in range(0, self.n_obj):
            f[:, i] = (1 + g)
            f[:, i] *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - i], alpha) * np.pi / 2.0), axis=1)
            if i > 0:
                f[:, i] *= np.sin(np.power(X_[:, X_.shape[1] - i], alpha) * np.pi / 2.0)

    def pareto_front(self):
        if self.n_pareto_points is None:
            raise Exception("Please specify how many pareto-optimal points are desired by setting n_pareto_points!")

        return super().pareto_front()


def generic_sphere(n_points, n_dim):
    w = get_uniform_weights(n_points, n_dim)
    F = w / np.tile(np.linalg.norm(w, axis=1)[:, None], (1, w.shape[1]))
    return F


class DTLZ1(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self):
        return 0.5 * get_uniform_weights(self.n_pareto_points, self.n_obj)

    def _evaluate(self, x, f):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        for i in range(0, self.n_obj):
            f[:, i] = 0.5 * (1 + g)
            f[:, i] *= np.prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                f[:, i] *= 1 - X_[:, X_.shape[1] - i]


class DTLZ2(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self):
        return generic_sphere(self.n_pareto_points, self.n_obj)

    def _evaluate(self, x, f):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        self.obj_func(X_, g, f, alpha=1)


class DTLZ3(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self):
        return generic_sphere(self.n_pareto_points, self.n_obj)

    def _evaluate(self, x, f):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        self.obj_func(X_, g, f, alpha=1)


class DTLZ4(DTLZ):
    def __init__(self, n_var=10, n_obj=3, alpha=100, d=100, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.alpha = alpha
        self.d = d

    def _calc_pareto_front(self):
        return generic_sphere(self.n_pareto_points, self.n_obj)

    def _evaluate(self, x, f):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        self.obj_func(X_, g, f, alpha=self.alpha)


class DTLZ5(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self):
        pass

    def _evaluate(self, x, f):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta[:, 0] = x[:, 0]
        self.obj_func(theta, g, f)


class DTLZ6(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self):
        pass

    def _evaluate(self, x, f):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = np.sum(np.power(X_M, 0.1), axis=1)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta[:, 0] = x[:, 0]
        self.obj_func(theta, g, f)


class DTLZ7(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _evaluate(self, x, f):
        for i in range(0, self.n_obj - 1):
            f[:, i] = x[:, i]

        g = 1 + 9 / self.k * np.sum(x[:, -self.k:], axis=1)
        h = self.n_obj - np.sum(f[:, :-1] / (1 + g[:, None]) * (1 + np.sin(3 * np.pi * f[:, :-1])), axis=1)
        f[:, self.n_obj - 1] = (1 + g) * h



