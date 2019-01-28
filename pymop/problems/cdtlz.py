import autograd.numpy as anp

from pymop import DTLZ3, DTLZ1


class C1DTLZ1(DTLZ1):

    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = 1

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        F = out["F"]
        out["G"] = - (1 - F[:, -1] / 0.6 - anp.sum(F[:, :-1] / 0.5, axis=1))


class C1DTLZ3(DTLZ3):

    def __init__(self, n_var=12, n_obj=3, r=None, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = 1

        if r is None:
            if self.n_obj < 5:
                r = 9.0
            elif 5 <= self.n_obj <= 12:
                r = 12.5
            else:
                r = 15.0

        self.r = r

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        _r = anp.sum(out["F"] ** 2, axis=1)
        out["G"] = - (_r - 4 ** 2) * (_r - self.r ** 2)
