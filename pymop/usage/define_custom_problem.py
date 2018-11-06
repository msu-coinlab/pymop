import numpy as np

from pymop.problem import Problem


# always derive from the main problem for the evaluation
class MyProblem(Problem):

    def __init__(self, var1=5, var2=0.1):
        super().__init__.__init__(self)

        # define the number of variables the problem has
        self.n_var = 10

        # define the number of constraints
        self.n_constr = 2

        # the number of objectives - just set it to 1 if single-objective
        self.n_obj = 1

        # define lower and upper bounds -  1d array with length equal to number of variable
        self.xl = -5 * np.ones(self.n_var)
        self.xu = 5 * np.ones(self.n_var)

        # store custom variables needed for evaluation
        self.var1 = var1
        self.var2 = var2

    # implemented the function evaluation function - the arrays to fill are provided directly
    def _evaluate(self, x, f, g, *args, **kwargs):
        # define an objective function to be evaluated using var1
        f[:, 0] = np.sum(np.power(x, 2) - self.var1 * np.cos(2 * np.pi * x), axis=1)

        # !!! only if a constraint value is positive it is violated !!!
        # set the constraint that x1 + x2 > var2
        g[:, 0] = (x[:, 0] + x[:, 1]) - self.var2

        # set the constraint that x3 + x4 < var2
        g[:, 1] = self.var2 - (x[:, 2] + x[:, 3])


problem = MyProblem()
F, G = problem.evaluate(np.random.rand(100, 10))
