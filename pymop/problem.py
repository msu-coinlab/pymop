import os
import numpy as np
import inspect


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

            if self._pareto_front is None:
                raise Exception("Pareto front for this tests problem not found.")

        return self._pareto_front

    def evaluate(self, X, return_constraint_violation=True, return_constraints=False, **kwargs):

        """
        Evaluate the given problem.

        The function values set as defined in the function.
        The constraint values are meant to be positive if infeasible. A higher positive values means "more" infeasible".
        If they are 0 or negative, they will be considered as feasible what ever their value is.

        Parameters
        ----------
        X : np.ndarray
            A two dimensional matrix where each row is a point to evaluate and each column a variable.

        return_constraint_violation : bool
            Whether the constraint violation is returned or not. If no constraint exists,
            an array with zero values is returned. Default: True.
            
        return_constraints : bool
            Whether all constraint values are returned or not. Default: False.

        Returns
        -------
        F : np.ndarray
            Objective Values
        CV : np.ndarray
            Constraint Violations as a one dimensional array.
        G : np.ndarray
            Constraints as a two dimensional array.

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
        params = inspect.signature(self.func).parameters

        # optionally allow to have individuals also as a parameter and call the subfunction not in kwargs
        if 'individuals' in kwargs:
            individuals = kwargs.pop('individuals')
        else:
            individuals = None

        _kwargs = {}
        for key, value in kwargs.items():
            if key in params:
                _kwargs[key] = value

        args = [X, f]

        if self.n_constr > 0:
            args.append(g)
            if len(params) == 4:
                args.append(individuals)
        else:
            if len(params) == 3:
                args.append(individuals)

        self.func(*args, **_kwargs)

        # create the returned values in a list
        vals = [f]
        if return_constraint_violation:
            vals.append(Problem.calc_constraint_violation(g))
        if return_constraints:
            vals.append(g)

        # convert back if just one vector is evaluated
        if only_single_value:
            vals = [e[0, :] for e in vals]

        if len(vals) > 1:
            return tuple(vals)
        else:
            return vals[0]

    def name(self):
        """
        Returns
        -------
        name : str
            The name of the problem. Per default it is the name of the class but it can be overridden.
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
        fname = os.path.join(current_dir, "pf", "%s.pf" % self.name())
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
        if G is None:
            return None
        elif G.shape[1] == 0:
            return np.zeros(G.shape[0])[:, None]
        else:
            return np.sum(G * (G > 0).astype(np.float), axis=1)[:, None]


class ScaledProblem(Problem):

    def __init__(self, problem, scale_factor):
        super().__init__(problem.n_var, problem.n_obj, problem.n_constr, problem.xl, problem.xu, problem.func)
        self.problem = problem
        self.scale_factor = scale_factor

    @staticmethod
    def get_scale(n, scale_factor):
        return np.power(np.full(n, scale_factor), np.arange(n))

    def evaluate(self, X, **kwargs):
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

    def evaluate(self, X, **kwargs):
        t = self.problem.evaluate(X, **kwargs)
        F = np.power(t[0], ConvexProblem.get_power(self.n_obj))
        return tuple([F] + list(t)[1:])

    def _calc_pareto_front(self):
        F = self.problem.pareto_front()
        return np.power(F, ConvexProblem.get_power(self.n_obj))
