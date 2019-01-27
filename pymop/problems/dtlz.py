import autograd.numpy as anp

from pymop.problem import Problem


class DTLZ(Problem):
    def __init__(self, n_var, n_obj, k=None):

        if n_var:
            self.k = n_var - n_obj + 1
        elif k:
            self.k = k
            n_var = k + n_obj - 1
        else:
            raise Exception("Either provide number of variables or k!")

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0, xu=1, type_var=anp.double)

    def g1(self, X_M):
        return 100 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(20 * anp.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return anp.sum(anp.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= anp.prod(anp.cos(anp.power(X_[:, :X_.shape[1] - i], alpha) * anp.pi / 2.0), axis=1)
            if i > 0:
                _f *= anp.sin(anp.power(X_[:, X_.shape[1] - i], alpha) * anp.pi / 2.0)

            f.append(_f)

        f = anp.column_stack(f)
        return f


def generic_sphere(ref_dirs):
    return ref_dirs / anp.tile(anp.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))


class DTLZ1(DTLZ):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs=None):
        return 0.5 * ref_dirs

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)

        f = []
        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= anp.prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        out["F"] = anp.column_stack(f)


class DTLZ2(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)


class DTLZ3(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)


class DTLZ4(DTLZ):
    def __init__(self, n_var=10, n_obj=3, alpha=100, d=100, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.alpha = alpha
        self.d = d

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=self.alpha)


class DTLZ5(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self):
        raise Exception("Not implemented yet.")

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = anp.column_stack([x[:, 0], theta[:, 1:]])

        out["F"] = self.obj_func(theta, g)


class DTLZ6(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self):
        raise Exception("Not implemented yet.")

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = anp.sum(anp.power(X_M, 0.1), axis=1)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = anp.column_stack([x[:, 0], theta[:, 1:]])

        out["F"] = self.obj_func(theta, g)


class DTLZ7(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(x[:, i])
        f = anp.column_stack(f)

        g = 1 + 9 / self.k * anp.sum(x[:, -self.k:], axis=1)
        h = self.n_obj - anp.sum(f / (1 + g[:, None]) * (1 + anp.sin(3 * anp.pi * f)), axis=1)

        out["F"] = anp.column_stack([f, (1 + g) * h])


class ScaledProblem(Problem):

    def __init__(self, problem, scale_factor):
        super().__init__(n_var=problem.n_var, n_obj=problem.n_obj, n_constr=problem.n_constr,
                         xl=problem.xl, xu=problem.xu, type_var=problem.type_var)
        self.problem = problem
        self.scale_factor = scale_factor

    @staticmethod
    def get_scale(n, scale_factor):
        return anp.power(anp.full(n, scale_factor), anp.arange(n))

    def evaluate(self, X, *args, **kwargs):
        t = self.problem.evaluate(X, **kwargs)
        F = t[0] * ScaledProblem.get_scale(self.n_obj, self.scale_factor)
        return tuple([F] + list(t)[1:])

    def _calc_pareto_front(self, *args, **kwargs):
        return self.problem.pareto_front(*args, **kwargs) * ScaledProblem.get_scale(self.n_obj, self.scale_factor)


class ConvexProblem(Problem):

    def __init__(self, problem):
        super().__init__(problem.n_var, problem.n_obj, problem.n_constr, problem.xl, problem.xu)
        self.problem = problem

    @staticmethod
    def get_power(n):
        p = anp.full(n, 4.0)
        p[-1] = 2.0
        return p

    def evaluate(self, X, *args, **kwargs):
        t = self.problem.evaluate(X, **kwargs)
        F = anp.power(t[0], ConvexProblem.get_power(self.n_obj))
        return tuple([F] + list(t)[1:])

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        F = self.problem.pareto_front(ref_dirs)
        return anp.power(F, ConvexProblem.get_power(self.n_obj))


def constraint_c1(f, r, dtlz_type):
    if dtlz_type == 3:
        radius = anp.sum(f ** 2, axis=1)
        g = - (radius - 16) * (radius - r ** 2)
    elif dtlz_type == 1:
        g = - (1 - f[:, -1] / 0.6 - anp.sum(f[:, :-1] / 0.5, axis=1))

    return g


def constraint_c2(f, r):
    n_obj = f.shape[1]

    v1 = anp.inf*anp.ones(f.shape[0])

    for i in range(n_obj):
        temp = (f[:, i] - 1)**2 + (anp.sum(f**2, axis=1)-f[:, i]**2) - r**2
        v1 = anp.minimum(temp.flatten(), v1)

    a = 1/anp.sqrt(n_obj)
    v2 = anp.sum((f-a)**2, axis=1)-r**2
    g = anp.minimum(v1, v2.flatten())

    return g


def constraint_c3(f, dtlz_type):  # M circle if DTLZ2,3,4 and linear otherwise
    n_obj = f.shape[1]
    g = [] # anp.zeros(f.shape)

    for i in range(n_obj):
        if dtlz_type == 1:
            _g = 1 - f[:, i] / 0.5 - (anp.sum(f, axis=1) - f[:, i])
        elif dtlz_type == 2 or dtlz_type == 3 or dtlz_type == 4:
            _g = 1 - f[:, i] ** 2 / 4 - (anp.sum(f ** 2, axis=1) - f[:, i] ** 2)
        else:
            raise Exception("DTLZ type not supported for C3 constrained problem")
        g.append(_g)
    return anp.column_stack(g)


def constraint_c4(f, r):  # cylindrical
    l = anp.mean(f, axis=1)
    l = anp.expand_dims(l, axis=1)
    g = -anp.sum(anp.power(f-l, 2), axis=1) + anp.power(r, 2)

    return g


class C1DTLZ1(DTLZ1):

    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = 1

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        F = out["F"]
        out["G"] = - (1 - F[:, -1] / 0.6 - anp.sum(F[:, :-1] / 0.5, axis=1))


class C1DTLZ3(DTLZ3):

    def __init__(self, r=None, n_var=12, n_obj=3, **kwargs):
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


class C2DTLZ2(DTLZ2):

    def __init__(self, r=None, n_var=12, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = 1

        if r is None:
            if n_obj == 2:
                r = 0.2
            elif n_obj == 3:
                r = 0.4
            else:
                r = 0.5

        self.r = r

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c2(out["F"], self.r)

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):

        F = super().pareto_front(ref_dirs, *args, **kwargs)
        G = constraint_c2(F, r=self.r)
        G[G <= 0] = 0
        if G.ndim > 1:
            G = anp.sum(G, axis=1)
        return F[G <= 0]


class C3DTLZ4(DTLZ4):

    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = 1

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c3(out["F"], 4)

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):

        F = super().pareto_front(ref_dirs, *args, **kwargs)
        a = anp.sqrt(anp.sum(F ** 2, 1) - 3 / 4 * anp.max(F ** 2, axis=1))
        a = anp.expand_dims(a, axis=1)
        a = anp.tile(a, [1, ref_dirs.shape[1]])
        F = F / a

        return F
