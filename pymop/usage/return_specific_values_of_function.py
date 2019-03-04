import numpy as np

from pymop import ZDT1
from pymop.problems.osy import OSY

problem = ZDT1()
X = np.random.random((100, problem.n_var))

# by default if the problem has no constraints just the function values are returned
F = problem.evaluate(X)

problem = OSY()
X = np.random.random((100, problem.n_var))

# by default if the problem has constraints both is returned
F, CV = problem.evaluate(X)

# you can also ask for more input
F, CV, G = problem.evaluate(X, return_values_of=["F", "CV", "G"])

# or also for less input
F = problem.evaluate(X, return_values_of=["F"])
