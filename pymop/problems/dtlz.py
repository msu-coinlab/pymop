import numpy as np

from pymop.problem import Problem
from pymop.util import get_uniform_weights


class DTLZ(Problem):
    def __init__(self, n_var, n_obj):
        Problem.__init__(self)
        self.n_obj = n_obj
        self.n_var = n_var
        self.n_constr = 0
        self.func = self._evaluate
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

        self.k = self.n_var - self.n_obj + 1

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


def generic_sphere(ref_dirs):
    return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))


class DTLZ1(DTLZ):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs=None):
        return 0.5 * ref_dirs

    def _evaluate(self, x, f, *args, **kwargs):
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

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, f, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        self.obj_func(X_, g, f, alpha=1)


class DTLZ3(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, f, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        self.obj_func(X_, g, f, alpha=1)


class DTLZ4(DTLZ):
    def __init__(self, n_var=10, n_obj=3, alpha=100, d=100, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.alpha = alpha
        self.d = d

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, f, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        self.obj_func(X_, g, f, alpha=self.alpha)


class DTLZ5(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self):
        raise Exception("Not implemented yet.")

    def _evaluate(self, x, f, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta[:, 0] = x[:, 0]
        self.obj_func(theta, g, f)


class DTLZ6(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self):
        raise Exception("Not implemented yet.")

    def _evaluate(self, x, f, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = np.sum(np.power(X_M, 0.1), axis=1)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta[:, 0] = x[:, 0]
        self.obj_func(theta, g, f)


class DTLZ7(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _evaluate(self, x, f, *args, **kwargs):
        for i in range(0, self.n_obj - 1):
            f[:, i] = x[:, i]

        g = 1 + 9 / self.k * np.sum(x[:, -self.k:], axis=1)
        h = self.n_obj - np.sum(f[:, :-1] / (1 + g[:, None]) * (1 + np.sin(3 * np.pi * f[:, :-1])), axis=1)
        f[:, self.n_obj - 1] = (1 + g) * h


class ScaledProblem(Problem):

    def __init__(self, problem, scale_factor):
        super().__init__(problem.n_var, problem.n_obj, problem.n_constr, problem.xl, problem.xu, problem.func)
        self.problem = problem
        self.scale_factor = scale_factor

    @staticmethod
    def get_scale(n, scale_factor):
        return np.power(np.full(n, scale_factor), np.arange(n))

    def evaluate(self, X, *args, **kwargs):
        t = self.problem.evaluate(X, **kwargs)
        F = t[0] * ScaledProblem.get_scale(self.n_obj, self.scale_factor)
        return tuple([F] + list(t)[1:])

    def _calc_pareto_front(self):
        return self.problem.pareto_front() * ScaledProblem.get_scale(self.n_obj, self.scale_factor)


class ConvexProblem(Problem):

    def __init__(self, problem):
        super().__init__(problem.n_var, problem.n_obj, problem.n_constr, problem.xl, problem.xu, problem.func)
        self.problem = problem

    @staticmethod
    def get_power(n):
        p = np.full(n, 4.0)
        p[-1] = 2.0
        return p

    def evaluate(self, X, *args, **kwargs):
        t = self.problem.evaluate(X, **kwargs)
        F = np.power(t[0], ConvexProblem.get_power(self.n_obj))
        return tuple([F] + list(t)[1:])

    def _calc_pareto_front(self):
        F = self.problem.pareto_front()
        return np.power(F, ConvexProblem.get_power(self.n_obj))
