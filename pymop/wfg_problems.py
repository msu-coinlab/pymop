import numpy as np
import optproblems.wfg as wfg
from pymop.problem import Problem

class WFG(Problem):
    def __init__(self, n_var, n_obj, k=None):
        Problem.__init__(self)
        self.n_obj = n_obj
        self.n_var = n_var
        self.k = 2 * (self.n_obj - 1) if k is None else k
        self.func = self._evaluate
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

class WFG1(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = wfg.WFG1(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)


class WFG2(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)


    def _evaluate(self, x, f):
        func = wfg.WFG2(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)


class WFG3(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = wfg.WFG3(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)


class WFG4(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = wfg.WFG4(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)


class WFG5(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = wfg.WFG5(num_variables=self.n_var, num_objectives=self.n_obj, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)


class WFG6(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = wfg.WFG6(num_variables=self.n_var, num_objectives=self.n_obj, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)


class WFG7(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = wfg.WFG7(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)

class WFG8(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = wfg.WFG8(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)

class WFG9(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = wfg.WFG9(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    func = wfg.WFG1(num_objectives=3, num_variables=12, k=4)
    s = func.get_optimal_solutions(1000)
    sol = []
    for n in range(len(s)):
        sol.append(func(s[n].phenome))
    sol = np.array(sol)
    fig = plt.figure()
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sol[:, 0], sol[:, 1], sol[:, 2])
    plt.show()

