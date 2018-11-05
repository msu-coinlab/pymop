
import numpy as np

# this will be the evaluation function that is called each time
from pymop.factory import get_problem_from_func


def my_evaluate_func(x):
    import numpy as np

    # number of inputs to calculate the objective for
    n = x.shape[0]

    # define output array for two objectives and one constraint
    f = np.full((n, 2), np.inf)
    g = np.full((n, 1), np.inf)

    # define the objective as x^2
    f[:, 0] = np.sum(np.square(x - 2))
    f[:, 1] = np.sum(np.square(x + 2))

    # x^2 < 2 constraint
    g[:, 0] = np.sum(np.square(x - 1))

    return f, g


# load the problem from a function - define 3 variables with the same lower bound
problem = get_problem_from_func(my_evaluate_func, -10, 10, n_var=3)
F, CV = problem.evaluate(np.random.rand(100, 3))

# or define a problem with varying lower and upper bounds
problem = get_problem_from_func(my_evaluate_func, np.array([-10, -5, -10]), np.array([10, 5, 10]))
F, CV = problem.evaluate(np.random.rand(100, 3))
