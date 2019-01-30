import autograd.numpy as anp
import matplotlib.pyplot as plt
from pymop.problems.dtlz import DTLZ, DTLZ1, DTLZ2, DTLZ3, DTLZ4, generic_sphere
from pymop.util import UniformReferenceDirectionFactory


class DTLZ4B(DTLZ):
    def __init__(self, n_var=10, n_obj=3, alpha=100, d=100, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.alpha = alpha
        self.d = d
    def g2(self, X_M):
        return 100*anp.sum(anp.square(X_M - 0.5), axis=1)

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=self.alpha)

class C1DTLZ1(DTLZ1):

    def __init__(self, n_var=12, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = 1

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c1_linear(out["F"])

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        return super()._calc_pareto_front(ref_dirs, *args, **kwargs)


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
        out["G"] = constraint_c1_spherical(out["F"], self.r)

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        return super()._calc_pareto_front(ref_dirs, *args, **kwargs)


class C2DTLZ2(DTLZ2):

    def __init__(self, n_var=12, n_obj=3, r=None, **kwargs):
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
        F = super()._calc_pareto_front(ref_dirs, *args, **kwargs)
        G = constraint_c2(F, r=self.r)
        G[G <= 0] = 0
        return F[G <= 0]


class C3DTLZ4(DTLZ4):

    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = n_obj

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c3_spherical(out["F"])

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        F = super()._calc_pareto_front(ref_dirs, *args, **kwargs)
        a = anp.sqrt(anp.sum(F ** 2, 1) - 3 / 4 * anp.max(F ** 2, axis=1))
        a = anp.expand_dims(a, axis=1)
        a = anp.tile(a, [1, ref_dirs.shape[1]])
        F = F / a

        return F


def constraint_c1_linear(f):

    g = - (1 - f[:, -1] / 0.6 - anp.sum(f[:, :-1] / 0.5, axis=1))

    return g


def constraint_c1_spherical(f, r):
    radius = anp.sum(f ** 2, axis=1)
    g = - (radius - 16) * (radius - r ** 2)

    return g


def constraint_c2(f, r):
    n_obj = f.shape[1]

    v1 = anp.inf * anp.ones(f.shape[0])

    for i in range(n_obj):
        temp = (f[:, i] - 1) ** 2 + (anp.sum(f ** 2, axis=1) - f[:, i] ** 2) - r ** 2
        v1 = anp.minimum(temp.flatten(), v1)

    a = 1 / anp.sqrt(n_obj)
    v2 = anp.sum((f - a) ** 2, axis=1) - r ** 2
    g = anp.minimum(v1, v2.flatten())

    return g


def constraint_c3_linear(f):  # M lines
    n_obj = f.shape[1]
    g = []

    for i in range(n_obj):
        _g = 1 - f[:, i] / 0.5 - (anp.sum(f, axis=1) - f[:, i])
        g.append(_g)

    return anp.column_stack(g)


def constraint_c3_spherical(f):  # M ellipse
    n_obj = f.shape[1]
    g = []

    for i in range(n_obj):
        _g = 1 - f[:, i] ** 2 / 4 - (anp.sum(f ** 2, axis=1) - f[:, i] ** 2)
        g.append(_g)
    return anp.column_stack(g)


def constraint_c4_cylindrical(f, r):  # cylindrical
    l = anp.mean(f, axis=1)
    l = anp.expand_dims(l, axis=1)
    g = -anp.sum(anp.power(f - l, 2), axis=1) + anp.power(r, 2)

    return g


class CDTLZ(DTLZ):

    def __init__(self, dtlz, clist=None, rlist=None, **kwargs):

        # assert len(clist) == len(rlist)
        clist = anp.asarray(clist)
        self.clist = anp.unique(clist)
        self.n_obj = dtlz.n_obj

        if rlist is None:
            rlist = []
            for i in range(len(self.clist)):
                if self.clist[i] == 1:
                    if self.n_obj < 5:
                        rlist.append(20.0)  # rlist.append(9.0)
                    elif 5 <= self.n_obj <= 12:
                        rlist.append(20.0)  # rlist.append(12.5)
                    elif self.n_obj <= 15:
                        rlist.append(20.0)  # rlist.append(15.0)
                    else:
                        raise Exception("Parameter r for C1 is not defined for Obj. "+self.n_obj)
                elif self.clist[i] == 2:
                    if self.n_obj == 2:
                        rlist.append(0.2)
                    elif self.n_obj == 3:
                        rlist.append(0.4)
                    else:
                        rlist.append(0.5)
                elif self.clist[i] == 4:
                    if self.n_obj == 2:
                        rlist.append(0.225)
                    elif self.n_obj == 3:
                        rlist.append(0.225)
                    elif self.n_obj == 5:
                        rlist.append(0.225)
                    elif self.n_obj == 8:
                        rlist.append(0.26)
                    elif self.n_obj == 10:
                        rlist.append(0.26)
                    elif self.n_obj == 15:
                        rlist.append(0.27)
                    else:
                        raise Exception("Parameter r for C4 is not defined for Obj. " + self.n_obj)
                else:
                    rlist.append(0)

        self.rlist = anp.asarray(rlist)
        self.dtlz = dtlz

        super().__init__(n_var=dtlz.n_var, n_obj=dtlz.n_obj, **kwargs)
        self.n_constr = len(anp.unique(clist))

        if isinstance(dtlz, DTLZ1):
            self.dtlz_type = 1
        elif isinstance(dtlz, DTLZ2):
            self.dtlz_type = 2
        elif isinstance(dtlz, DTLZ3):
            self.dtlz_type = 3
        elif isinstance(dtlz, DTLZ4) or isinstance(dtlz, DTLZ4B):
            self.dtlz_type = 4
        else:
            raise Exception("DTLZ problem not supported.")

    def _evaluate(self, X, out, *args, **kwargs):

        # merge the result we get from the dtlz
        self.dtlz._evaluate(X, out, *args, **kwargs)
        g = []
        for i in range(self.n_constr):
            if self.clist[i] == 1:
                if self.dtlz_type == 1:
                    _g = constraint_c1_linear(out["F"])
                else:
                    _g = constraint_c1_spherical(out["F"], self.rlist[i])
            elif self.clist[i] == 2:
                _g = constraint_c2(out["F"], self.rlist[i])
            elif self.clist[i] == 3:
                if self.dtlz_type == 1:
                    _g = constraint_c3_linear(out["F"])
                else:
                    _g = constraint_c3_spherical(out["F"])
            elif self.clist[i] == 4:
                _g = constraint_c4_cylindrical(out["F"], self.rlist[i])
            else:
                _g = []

            g.append(_g)

        out["G"] = anp.column_stack(g)

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):

        F = self.dtlz.pareto_front(ref_dirs, *args, **kwargs)

        if anp.any(self.clist == 3) and not anp.any(self.clist == 2):
            if self.dtlz_type == 1:
                F = 0.5 * ref_dirs
            else:
                a = anp.sqrt(anp.sum(F ** 2, 1) - 3 / 4 * anp.max(F ** 2, axis=1))
                a = anp.expand_dims(a, axis=1)
                a = anp.tile(a, [1, ref_dirs.shape[1]])
                F = F / a

                if anp.any(self.clist == 4):  # 3 and 4 together
                    r = self.rlist[anp.where(self.clist == 4)][0]
                    _g = constraint_c4_cylindrical(F, r)
                    _g[_g <= 0] = 0
                    F = F[_g <= 0]

        elif anp.any(self.clist == 2):  # C2 cannot be with C3
            r = self.rlist[anp.where(self.clist == 2)][0]
            _g = constraint_c2(F, r)
            _g[_g <= 0] = 0
            F = F[_g <= 0]
            if anp.any(self.clist == 4):  # 2 and 4 together
                r = self.rlist[anp.where(self.clist == 4)][0]
                _g = constraint_c4_cylindrical(F, r)
                _g[_g <= 0] = 0
                F = F[_g <= 0]
        elif anp.any(self.clist == 4):
            r = self.rlist[anp.where(self.clist == 4)][0]
            _g = constraint_c4_cylindrical(F, r)
            _g[_g <= 0] = 0
            F = F[_g <= 0]

        return F


problems = [('CDTLZ', [DTLZ3(n_var=2, n_obj=2), [1]]),
            ('CDTLZ', [DTLZ3(n_var=2, n_obj=2), [1, 2]]),
            ('CDTLZ', [DTLZ3(n_var=2, n_obj=2), [1, 3, 4]]),
            ('CDTLZ', [DTLZ3(n_var=2, n_obj=2), [3, 4]]),
            ('CDTLZ', [DTLZ4B(n_var=2, n_obj=2), [1, 4]]),
            ('CDTLZ', [DTLZ4B(n_var=2, n_obj=2), [1, 3, 4]]),
            ('CDTLZ', [DTLZ1(n_var=2, n_obj=2), [1]]),
            ]

if __name__ == '__main__':

    for prob_no, entry in enumerate(problems):
        name, params = entry

        problem = globals()[name](*params)
        print("Plotting: " + problem.dtlz.name())
        if isinstance(problem.dtlz, DTLZ1):
            dtlz_type = 1
        elif isinstance(problem.dtlz, DTLZ2):
            dtlz_type = 2
        elif isinstance(problem.dtlz, DTLZ3):
            dtlz_type = 3
        elif isinstance(problem.dtlz, DTLZ4) or isinstance(problem.dtlz, DTLZ4B):
            dtlz_type = 4
        else:
            raise Exception("DTLZ problem not supported.")

        nx = 100

        x = anp.linspace(0, 1, nx)

        if dtlz_type == 1:
            # y = np.linspace(0.5 - 7.5 * 1e-2, 0.5 + 7.5 * 1e-2, nx)  # DTLZ1
            y = anp.linspace(0.5 - 7.5 * 1e-4, 0.5 + 7.5 * 1e-4, nx)  # DTLZ1
        elif dtlz_type == 2:
            y = anp.linspace(0.5 - 0.2, 0.5 + 0.2, nx)
        elif dtlz_type == 3:
            y = anp.linspace(0.5 - 7.5 * 1e-3, 0.5 + 7.5 * 1e-3, nx)  # DTLZ3
        elif dtlz_type == 4:
            y = anp.linspace(0.5 - 0.4, 0.5 + 0.4, nx)

        xv, yv = anp.meshgrid(x, y)
        x = anp.vstack(xv.flatten())
        y = anp.vstack(yv.flatten())
        X = anp.hstack((x, y))
        n = x.shape[0]

        out = dict()
        G = anp.zeros((n, problem.n_constr))
        problem._evaluate(X, out)
        F = out["F"]
        G = out["G"]

        F1 = F[:, 0].flatten()
        F2 = F[:, 1].flatten()

        G[G <= 0] = 0
        if G.ndim > 1:
            G = anp.sum(G, axis=1)
        index = G > 0

        ref_dirs = UniformReferenceDirectionFactory(n_dim=problem.n_obj, n_points=21).do()
        PF = problem._calc_pareto_front(ref_dirs)

        plt.clf()
        plt.title(problem.dtlz.name())
        plt.plot(F1, F2, 'o', color='#DCDCDC')  # '#A9A9A9'
        plt.plot(F1[index], F2[index], 'o', color='b')
        plt.plot(PF[:, 0], PF[:, 1], 'o', color='r')

        plt.show()

