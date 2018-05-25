"""
This is the base class for a problem is used to define specific problem definitions. 
If for the given problem the optima are known they can be provided by overwriting the given functions.
"""

import numpy as np

class Problem:
    def __init__(self, n_var=0, n_obj=0, n_constr=0, xl=None, xu=None, func=None):
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.xl = xl if type(xl) is np.ndarray else np.ones(n_var) * xl
        self.xu = xu if type(xu) is np.ndarray else np.ones(n_var) * xu
        self.func = func
        self._pareto_front = None

    # return the maximum objective values of the pareto front
    def nadir_point(self):
        return np.max(self.pareto_front(), axis=0)

    # return the minimum values of the pareto front
    def ideal_point(self):
        return np.min(self.pareto_front(), axis=0)

    # return the pareto front
    def pareto_front(self):
        if self._pareto_front is None:
            self._pareto_front = self.calc_pareto_front()
        return self._pareto_front

    """
    Evaluate the given problem.
    
    The function values set as defined in the function. 
    The constraint values are meant to be positive if infeasible. A higher positive values means "more" infeasible".
    If they are 0 or negative, they will be considered as feasible what ever their value is.
    
    return_constraints: 0 - No constraints are returned 
                        1 - All constraints 
                        2 - Only constraint violations (vector of zeros if problem has no constraints)
                           
    """

    def evaluate(self, x, return_constraints=1):

        only_single_value = len(np.shape(x)) == 1
        if only_single_value:
            x = np.array([x])

        # check the dimensionality of the problem and the given input
        if x.shape[1] != self.n_var:
            raise Exception('Input dimension %s are not equal to n_var %s!' % (x.shape[1], self.n_var))

        # create the resulting arrays
        f = np.zeros((x.shape[0], self.n_obj))
        g = np.zeros((x.shape[0], self.n_constr))

        # if constraints exists func(x, f, g) is used otherwise just func(x, f)
        if self.n_constr > 0:
            self.func(x, f, g)
        else:
            self.func(x, f)

        # convert back if just one vector is evaluated
        if only_single_value:
            return f[0, :], g[0, :]

        if return_constraints == 0:
            return f
        elif return_constraints == 1:
            return f, g
        elif return_constraints == 2:
            return f, Problem.calc_constraint_violation(g)

    # name of the problem
    def name(self):
        return self.__class__.__name__

    # some problem information
    def __str__(self):
        s = "# name: %s\n" % self.name()
        s += "# n_var: %s\n" % self.n_var
        s += "# n_obj: %s\n" % self.n_obj
        s += "# n_constr: %s\n" % self.n_constr
        s += "# f(xl): %s\n" % self.evaluate(self.xl)[0]
        s += "# f((xl+xu)/2): %s\n" % self.evaluate((self.xl + self.xu) / 2.0)[0]
        s += "# f(xu): %s\n" % self.evaluate(self.xu)[0]
        return s

    @staticmethod
    def calc_constraint_violation(G):
        if G.shape[1] == 0:
            return np.zeros(G.shape[0])
        else:
            return np.sum(G * (G > 0).astype(np.float), axis=1)[:, None]


if __name__ == "__main__":

    # numpy arrays are required as an input
    import numpy as np

    # first import the specific problem to be solved
    from pymop.dtlz import DTLZ1

    # initialize it with the necessary parameters
    problem = DTLZ1(n_var=10, n_obj=3)

    # evaluation function returns by default two numpy arrays - objective function values and constraints -
    # as input either provide a vector
    F, G = problem.evaluate(np.random.random(10))

    # or a whole matrix to evaluate several solutions at once
    F, G = problem.evaluate(np.random.random((100, 10)))

    # if no constraints should be returned
    F = problem.evaluate(np.random.random((100, 10)), return_constraints=0)

    # if only the constraint violation should be returned - vector of zeros if no constraints exist
    from pymop.welded_beam import WeldedBeam

    problem = WeldedBeam()
    F, CV = problem.evaluate(np.random.random((100, 4)), return_constraints=2)
