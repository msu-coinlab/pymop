import autograd.numpy as anp
import numpy as np

from pymop.problem import Problem


class ZDT(Problem):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0, xu=1, type_var=anp.double, **kwargs)


class ZDT1(ZDT):

    def _calc_pareto_front(self, n_pareto_points=100):
        x = anp.linspace(0, 1, n_pareto_points)
        return anp.array([x, 1 - anp.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * anp.sum(x[:, 1:], axis=1)
        f2 = g * (1 - anp.power((f1 / g), 0.5))

        out["F"] = anp.column_stack([f1, f2])

        if "dF" in out:
            dF = np.zeros([x.shape[0], self.n_obj, self.n_var], dtype=np.float)
            dF[:, 0, 0], dF[:, 0, 1:] = 1, 0
            dF[:, 1, 0] = -0.5 * anp.sqrt(g / x[:, 0])
            dF[:, 1, 1:] = ((9 / (self.n_var - 1)) * (1 - 0.5 * anp.sqrt(x[:, 0] / g)))[:, None]
            out["dF"] = dF


class ZDT2(ZDT):

    def _calc_pareto_front(self, n_pareto_points=100):
        x = anp.linspace(0, 1, n_pareto_points)
        return anp.array([x, 1 - anp.power(x, 2)]).T

    def _evaluate(self, x, f, *args, **kwargs):
        f[:, 0] = x[:, 0]
        c = anp.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f[:, 1] = g * (1 - anp.power((f[:, 0] * 1.0 / g), 2))

    def _grad(self, x, df, *args, **kwargs):
        g = 1 + 9.0 / (self.n_var - 1) * anp.sum(x[:, 1:], axis=1)

        df[:, 0, 0], df[:, 0, 1:] = 1, 0
        df[:, 1, 0] = -2 * x[:, 0] / g
        df[:, 1, 1:] = (9 / (self.n_var - 1)) * (1 + x[:, 0] ** 2 / g ** 2)[:, None]


class ZDT3(ZDT):

    def _calc_pareto_front(self, n_pareto_points=100):
        regions = [[0, 0.0830015349],
                   [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041],
                   [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]

        pareto_front = anp.array([]).reshape((-1, 2))
        for r in regions:
            x1 = anp.linspace(r[0], r[1], int(n_pareto_points / len(regions)))
            x2 = 1 - anp.sqrt(x1) - x1 * anp.sin(10 * anp.pi * x1)
            pareto_front = anp.concatenate((pareto_front, anp.array([x1, x2]).T), axis=0)
        return pareto_front

    def _evaluate(self, x, f, *args, **kwargs):
        f[:, 0] = x[:, 0]
        c = anp.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f[:, 1] = g * (1 - anp.power(f[:, 0] * 1.0 / g, 0.5) - (f[:, 0] * 1.0 / g) * anp.sin(10 * anp.pi * f[:, 0]))

    def _grad(self, x, df, *args, **kwargs):
        c = anp.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)

        df[:, 0, 0], df[:, 0, 1:] = 1, 0
        df[:, 1, 0] = -0.5 * anp.sqrt(g / x[:, 0]) - anp.sin(10 * anp.pi * x[:, 0]) - 10 * anp.pi * x[:, 0] * anp.cos(
            10 * anp.pi * x[:, 0])
        df[:, 1, 1:] = (9 / (self.n_var - 1)) * (1 - 0.5 * anp.sqrt(x[:, 0] / g))[:, None]


class ZDT4(ZDT):
    def __init__(self, n_var=10):
        super().__init__(n_var)
        self.xl = -5 * anp.ones(self.n_var)
        self.xl[0] = 0.0
        self.xu = 5 * anp.ones(self.n_var)
        self.xu[0] = 1.0
        self.func = self._evaluate

    def _calc_pareto_front(self, n_pareto_points=100):
        x = anp.linspace(0, 1, n_pareto_points)
        return anp.array([x, 1 - anp.sqrt(x)]).T

    def _evaluate(self, x, f, *args, **kwargs):
        f[:, 0] = x[:, 0]
        g = 1.0
        g += 10 * (self.n_var - 1)
        for i in range(1, self.n_var):
            g += x[:, i] * x[:, i] - 10.0 * anp.cos(4.0 * anp.pi * x[:, i])
        h = 1.0 - anp.sqrt(f[:, 0] / g)
        f[:, 1] = g * h


class ZDT6(ZDT):

    def _calc_pareto_front(self, n_pareto_points=100):
        x = anp.linspace(0.2807753191, 1, n_pareto_points)
        return anp.array([x, 1 - anp.power(x, 2)]).T

    def _evaluate(self, x, f, *args, **kwargs):
        f[:, 0] = 1 - anp.exp(-4 * x[:, 0]) * anp.power(anp.sin(6 * anp.pi * x[:, 0]), 6)
        g = 1 + 9.0 * anp.power(anp.sum(x[:, 1:], axis=1) / (self.n_var - 1.0), 0.25)
        f[:, 1] = g * (1 - anp.power(f[:, 0] / g, 2))
