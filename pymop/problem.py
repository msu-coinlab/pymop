import os

import numpy as np


class Problem:
    """
    Superclass for each problem that is defined. It provides attributes such
    as the number of variables, number of objectives or constraints.
    Also, the lower and upper bounds are stored. If available the Pareto-front, nadir point
    and ideal point are stored.
    """

    def __init__(self, n_var=0, n_obj=0, n_constr=0, xl=None, xu=None, func=None):
        """

        Parameters
        ----------
        n_var : int
            number of variables
        n_obj : int
            number of objectives
        n_constr : int
            number of constraints
        xl : np.ndarray
            lower bounds for the variables
        xu : np.ndarray
            upper bounds for the variable
        func : func
            function that evaluates the problem. Useful if a problem is defined inplace.
        """
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.xl = xl if type(xl) is np.ndarray else np.ones(n_var) * xl
        self.xu = xu if type(xu) is np.ndarray else np.ones(n_var) * xu
        self.func = func
        self._pareto_front = None

    # return the maximum objective values of the pareto front
    def nadir_point(self):
        """
        Returns
        -------
        nadir_point : np.ndarray
            The nadir point for a multi-objective problem.
            If single-objective, it returns the best possible solution which is equal to the ideal point.

        """
        return np.max(self.pareto_front(), axis=0)

    # return the minimum values of the pareto front
    def ideal_point(self):
        """
        Returns
        -------
        ideal_point : np.ndarray
            The ideal point for a multi-objective problem. If single-objective
            it returns the best possible solution.
        """
        return np.min(self.pareto_front(), axis=0)

    def pareto_front(self):
        """
        Returns
        -------
        P : np.ndarray
            The Pareto front of a given problem. It is only loaded or calculate the first time and then cached.
            For a single-objective problem only one point is returned but still in a two dimensional array.
        """
        if self._pareto_front is None:
            self._pareto_front = self._calc_pareto_front()
        return self._pareto_front


    def evaluate(self, X, return_constraints=1):

        """
        Evaluate the given problem.

        The function values set as defined in the function.
        The constraint values are meant to be positive if infeasible. A higher positive values means "more" infeasible".
        If they are 0 or negative, they will be considered as feasible what ever their value is.

        Parameters
        ----------
        X : np.ndarray
            A two dimensional matrix where each row is a point to evaluate and each column a variable.

        return_constraints : int
                        | 0 - No constraints are returned
                        | 1 - All constraints
                        | 2 - Only constraint violations (vector of zeros if problem has no constraints)

        Returns
        -------
        F : np.ndarray
            Objective Values
        G : np.ndarray
            Constraint Values. Depending on return_constraints only CV or not at all.

        """


        only_single_value = len(np.shape(X)) == 1
        if only_single_value:
            X = np.array([X])

        # check the dimensionality of the problem and the given input
        if X.shape[1] != self.n_var:
            raise Exception('Input dimension %s are not equal to n_var %s!' % (X.shape[1], self.n_var))

        # create the resulting arrays
        f = np.zeros((X.shape[0], self.n_obj))
        g = np.zeros((X.shape[0], self.n_constr))

        # if constraints exists func(x, f, g) is used otherwise just func(x, f)
        if self.n_constr > 0:
            self.func(X, f, g)
        else:
            self.func(X, f)

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
        """
        Returns
        -------
        name : str
            The name of the problem. Per default it is the name of the class but it can be overriden.
        """
        return self.__class__.__name__

    def _calc_pareto_front(self):
        """
        Default behaviour is look to look for the pareto front file.
        If this does not exist return None.

        Returns
        -------
        pf : numpy.array
            Pareto optimal front for the problem. In case of single-objective just one value

        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        fname = os.path.join(current_dir, "pf", "%s.pf" % self.__class__.__name__)
        if os.path.isfile(fname):
            return np.loadtxt(fname)
        return None

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
            return np.zeros(G.shape[0])[:, None]
        else:
            return np.sum(G * (G > 0).astype(np.float), axis=1)[:, None]


if __name__ == "__main__":

    from pymop.kursawe import Kursawe
    k = Kursawe()

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
