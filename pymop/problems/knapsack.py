import numpy as np

from pymop.problem import Problem


class Knapsack(Problem):
    def __init__(self,
                 n_items,  # number of items that can be picked up
                 W,  # weights for each item
                 P,  # profit of each item
                 C,  # maximum capacity
                 ):
        Problem.__init__(self)
        self.n_var = n_items
        self.n_constr = 1
        self.n_obj = 1
        self.func = self._evaluate

        self.W = W
        self.P = P
        self.C = C

    def _evaluate(self, x, f, g):
        g[:, 0] = np.sum(self.W * x, axis=1) - self.C
        f[:, 0] = -np.sum(self.P * x, axis=1)


def create_random_knapsack_problem(n_items):
    P = np.random.randint(1, 100, size=n_items)
    W = np.random.randint(1, 100, size=n_items)
    C = np.sum(W) / 2
    problem = Knapsack(n_items, W, P, C)
    return problem
