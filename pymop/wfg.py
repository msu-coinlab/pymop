import numpy as np
import optproblems

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

        self.func = None

    def pareto_front(self):
        n_optimal_solution = 1000
        pf = np.zeros((n_optimal_solution, self.n_obj))

        s = self.func.get_optimal_solutions(n_optimal_solution)
        for n in range(len(s)):
            pf[n,:] = self.func(s[n].phenome)
        return pf


class WFG1(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)
        self.func = optproblems.wfg.WFG1(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)

    def _evaluate(self, x, f):
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)

class WFG2(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)


    def _evaluate(self, x, f):
        func = optproblems.wfg.WFG2(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)


class WFG3(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = optproblems.wfg.WFG3(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)


class WFG4(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = optproblems.wfg.WFG4(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)


class WFG5(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = optproblems.wfg.WFG5(num_variables=self.n_var, num_objectives=self.n_obj, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)


class WFG6(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = optproblems.wfg.WFG6(num_variables=self.n_var, num_objectives=self.n_obj, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)


class WFG7(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = optproblems.wfg.WFG7(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)

class WFG8(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = optproblems.wfg.WFG8(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)

class WFG9(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(n_var, n_obj, k=k)

    def _evaluate(self, x, f):
        func = optproblems.wfg.WFG9(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = func(z)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    problem = WFG1(n_var=12, n_obj=3, k=4)
    pf = problem.pareto_front()

    fig = plt.figure()
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2])
    plt.show()

