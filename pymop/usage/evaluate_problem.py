import numpy as np

from pymop.problems.zdt import ZDT1

problem = ZDT1(n_var=10)

# evaluation function returns by default two numpy arrays - objective function values and constraints -
# as input either provide a vector
F, G = problem.evaluate(np.random.random(10))

# or a whole matrix to evaluate several solutions at once
F, G = problem.evaluate(np.random.random((100, 10)))

from pymop.problems.welded_beam import WeldedBeam
problem = WeldedBeam()

# if no constraints should be returned
F = problem.evaluate(np.random.random((100, 4)), return_values_of=["F"])

F, G, CV = problem.evaluate(np.random.random((100, 4)), return_values_of=["F", "G", "CV"])
