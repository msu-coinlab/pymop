import numpy as np

from pymop.problems.zdt import ZDT1

problem = ZDT1(n_var=10)

# evaluation function returns by default two numpy arrays - objective function values and constraints -
# as input either provide a vector
F, G = problem.evaluate(np.random.random(10), individuals=None)

# or a whole matrix to evaluate several solutions at once
F, G = problem.evaluate(np.random.random((100, 10)))

# if no constraints should be returned
F = problem.evaluate(np.random.random((100, 10)), return_constraint_violation=False)

from pymop.problems.welded_beam import WeldedBeam
F, CV = WeldedBeam().evaluate(np.random.random((100, 4)), return_constraint_violation=True, individuals=None)
