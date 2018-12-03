from abc import abstractmethod

import numpy as np


class Problem:
    """
    Superclass for each problem that is defined. It provides attributes such
    as the number of variables, number of objectives or constraints.
    Also, the lower and upper bounds are stored. If available the Pareto-front, nadir point
    and ideal point are stored.
    """

    def __init__(self, n_var=-1, n_obj=-1, n_constr=0, xl=None, xu=None, type_var=np.double):
        """

        Parameters
        ----------
        n_var : int
            number of variables
        n_obj : int
            number of objectives
        n_constr : int
            number of constraints
        xl : np.array or int
            lower bounds for the variables. if integer all lower bounds are equal.
        xu : np.array or int
            upper bounds for the variable. if integer all upper bounds are equal.
        type_var : numpy type
            type of the variable to be evaluated. Can also be np.object if it is a complex data type
        """

        # number of variable for this problem
        self.n_var = n_var

        # type of the variable to be evaluated
        self.type_var = type_var

        # number of objectives
        self.n_obj = n_obj

        # number of constraints
        self.n_constr = n_constr

        # allow just an integer for xl and xu if all bounds are equal
        if n_var > 0 and isinstance(xl, int) and isinstance(xu, int):
            self.xl = xl if type(xl) is np.ndarray else np.ones(n_var) * xl
            self.xu = xu if type(xu) is np.ndarray else np.ones(n_var) * xu
        else:
            self.xl = xl
            self.xu = xu

        # the pareto front will be calculated only once and is stored here
        self._pareto_front = None

    # return the maximum objective values of the pareto front
    def nadir_point(self):
        """
        Returns
        -------
        nadir_point : np.array
            The nadir point for a multi-objective problem.
            If single-objective, it returns the best possible solution which is equal to the ideal point.

        """
        return np.max(self.pareto_front(), axis=0)

    # return the minimum values of the pareto front
    def ideal_point(self):
        """
        Returns
        -------
        ideal_point : np.array
            The ideal point for a multi-objective problem. If single-objective
            it returns the best possible solution.
        """
        return np.min(self.pareto_front(), axis=0)

    def pareto_front(self, *args, **kwargs):
        """
        Returns
        -------
        P : np.array
            The Pareto front of a given problem. It is only loaded or calculate the first time and then cached.
            For a single-objective problem only one point is returned but still in a two dimensional array.
        """
        if self._pareto_front is None:
            self._pareto_front = self._calc_pareto_front(*args, **kwargs)

        return self._pareto_front

    def evaluate(self, X, *args,
                 return_constraint_violation=True,
                 return_constraints=False,
                 check_var_type=False,
                 **kwargs):

        """
        Evaluate the given problem.

        The function values set as defined in the function.
        The constraint values are meant to be positive if infeasible. A higher positive values means "more" infeasible".
        If they are 0 or negative, they will be considered as feasible what ever their value is.

        Parameters
        ----------
        X : np.array
            A two dimensional matrix where each row is a point to evaluate and each column a variable.

        return_constraint_violation : bool
            Whether the constraint violation is returned or not. If no constraint exists,
            an array with zero values is returned. Default: True.
            
        return_constraints : bool
            Whether all constraint values are returned or not. Default: False.

        check_var_type : bool
            Whether data types are checked to match or not. Might be desired if input is boolean and
            vector is real. However, lazy behaviour to just treat the input as it is, is default.

        Returns
        -------
        F : np.array
            Objective Values
        CV : np.array
            Constraint Violations as a two dimensional array.
        G : np.array
            Constraints as a two dimensional array.

        """

        only_single_value = len(np.shape(X)) == 1
        if only_single_value:
            X = np.array([X])

        if isinstance(X, np.ndarray):
            type_of_var = X.dtype
        else:
            type_of_var = type(X)

        if check_var_type and type_of_var != self.type_var:
            raise Exception('As variable type for this problem %s was defined. However, it is evaluated with %s!'
                            % (self.type_var, type_of_var))

        # check the dimensionality of the problem and the given input
        if X.shape[1] != self.n_var:
            raise Exception('Input dimension %s are not equal to n_var %s!' % (X.shape[1], self.n_var))

        # create the objective value array
        F = np.zeros((X.shape[0], self.n_obj))

        # create the constraint array and add to params
        G = np.zeros((X.shape[0], self.n_constr))

        if self.n_constr > 0:
            args = [G] + list(args)

        # call the function to evaluate
        self._evaluate(X, F, *args, **kwargs)

        # create the returned values in a list
        vals = [F]
        if return_constraint_violation:
            vals.append(Problem.calc_constraint_violation(G))
        if return_constraints:
            vals.append(G)

        # convert back if just one vector is evaluated
        if only_single_value:
            vals = [e[0, :] for e in vals]

        if len(vals) > 1:
            return tuple(vals)
        else:
            return vals[0]

    @abstractmethod
    def _evaluate(self, x, f, *args, **kwargs):
        pass

    def name(self):
        """
        Returns
        -------
        name : str
            The name of the problem. Per default it is the name of the class but it can be overridden.
        """
        return self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        """
        Method that either loads or calculates the pareto front. This is only done
        ones and the pareto front is stored.

        Returns
        -------
        pf : np.array
            Pareto front as array.

        """
        pass

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
